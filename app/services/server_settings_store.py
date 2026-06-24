"""Secure multi-profile store for local server export settings.

Storage layout:
  settings_dir/server_profiles_index.json   — plain JSON index (names only, no secrets)
  settings_dir/server_profiles/<safe>.enc   — Fernet-encrypted settings JSON per profile

On macOS (when Xcode CLT / swiftc is available):
  Secrets are stored in macOS Keychain with kSecAccessControlUserPresence.
  Every read triggers a native Touch ID / Face ID / password prompt automatically.
  PIN is not needed — the OS handles authentication.
  Binary: ~/.cache/face-local/keychain_helper  (compiled once)

On Windows:
  Secrets stored in Windows Credential Manager (DPAPI, user-account-bound).
  App-level PIN or Windows Hello used as an additional gate before reading.

On Linux / macOS without swiftc:
  Falls back to Python keyring + app-level PIN.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

_KEYRING_SERVICE = "face-local-server"
_INDEX_FILENAME = "server_profiles_index.json"
_PROFILES_SUBDIR = "server_profiles"
_CACHE_DIR = Path.home() / ".cache" / "face-local"
_ERR_MISSING_ENTITLEMENT = -34018  # errSecMissingEntitlement

# ---------------------------------------------------------------------------
# Swift helper: keychain_helper
#   Compiled once. Handles set / get / delete of keychain items protected by
#   kSecAccessControlUserPresence — every "get" triggers Touch ID / Face ID.
#   The secret value is passed via stdin (set) / stdout (get) to keep it out
#   of the process argument list.
# ---------------------------------------------------------------------------
_SWIFT_KEYCHAIN_HELPER = r"""
import Foundation
import Security
import LocalAuthentication

func die(_ msg: String) -> Never {
    fputs("ERROR: \(msg)\n", stderr)
    exit(1)
}

let args = CommandLine.arguments
guard args.count >= 4 else {
    die("usage: keychain_helper <set|get|delete> <service> <account> [reason]")
}

let mode    = args[1]
let service = args[2]
let account = args[3]

switch mode {

// ── set ───────────────────────────────────────────────────────────────────
case "set":
    // Read secret from stdin (never passes through process arg list)
    let secretData = FileHandle.standardInput.readDataToEndOfFile()
    guard !secretData.isEmpty else { die("no data on stdin") }

    var cfErr: Unmanaged<CFError>?
    guard let access = SecAccessControlCreateWithFlags(
        kCFAllocatorDefault,
        kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        .userPresence,   // every read requires Touch ID / Face ID / device password
        &cfErr
    ) else { die("SecAccessControlCreateWithFlags: \(cfErr!.takeRetainedValue())") }

    // Delete any previous item first (idempotent)
    let delQ: [CFString: Any] = [kSecClass: kSecClassGenericPassword,
                                  kSecAttrService: service,
                                  kSecAttrAccount: account]
    SecItemDelete(delQ as CFDictionary)

    let addQ: [CFString: Any] = [kSecClass:             kSecClassGenericPassword,
                                  kSecAttrService:        service,
                                  kSecAttrAccount:        account,
                                  kSecValueData:          secretData,
                                  kSecAttrAccessControl:  access]
    let st = SecItemAdd(addQ as CFDictionary, nil)
    guard st == errSecSuccess else { die("SecItemAdd \(st)") }
    exit(0)

// ── get ───────────────────────────────────────────────────────────────────
case "get":
    // Use modern LAContext API (kSecUseOperationPrompt deprecated in macOS 11)
    let reason = args.count >= 5 ? args[4] : "Face Recognizer"
    let ctx = LAContext()
    ctx.localizedReason = reason          // shown in Touch ID / Face ID / password prompt
    ctx.localizedFallbackTitle = ""       // hide "Enter Password" fallback button (OS shows it anyway)
    let query: [CFString: Any] = [kSecClass:                    kSecClassGenericPassword,
                                   kSecAttrService:              service,
                                   kSecAttrAccount:              account,
                                   kSecReturnData:               kCFBooleanTrue!,
                                   kSecUseAuthenticationContext: ctx]
    var ref: AnyObject?
    let st = SecItemCopyMatching(query as CFDictionary, &ref)
    switch st {
    case errSecSuccess:
        if let d = ref as? Data { FileHandle.standardOutput.write(d); exit(0) }
        die("unexpected result type")
    case errSecUserCanceled, -128:
        exit(2)  // user cancelled — caller can offer alternative
    case errSecItemNotFound:
        exit(3)  // no item stored yet
    default:
        die("SecItemCopyMatching \(st)")
    }

// ── delete ────────────────────────────────────────────────────────────────
case "delete":
    let delQ: [CFString: Any] = [kSecClass:       kSecClassGenericPassword,
                                  kSecAttrService: service,
                                  kSecAttrAccount: account]
    SecItemDelete(delQ as CFDictionary)
    exit(0)

default:
    die("unknown mode '\(mode)'")
}
"""


@dataclass
class ServerProfile:
    name: str
    admin_email: str = ""
    gmail_app_password: str = ""
    allowed_emails_csv: str = ""
    port: int = 3000
    domain: str = ""
    https_mode: str = "none"
    has_pin: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ServerProfile":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ProfileMeta:
    name: str
    has_pin: bool = False


class ServerSettingsStore:
    """Encrypted multi-profile store.

    On macOS (swiftc available): Keychain with kSecAccessControlUserPresence —
    every load auto-triggers Touch ID / Face ID / password via the OS.
    On Windows/Linux: keyring (DPAPI / Secret Service) + optional PIN gate.
    """

    def __init__(self, settings_dir: Optional[Path] = None) -> None:
        from app.paths import ensure_settings_dir
        self._dir = (settings_dir or ensure_settings_dir()).resolve()
        self._profiles_dir = self._dir / _PROFILES_SUBDIR
        self._index_file = self._dir / _INDEX_FILENAME
        self._lock = threading.Lock()
        self._compile_lock = threading.Lock()

        # macOS: kSecAccessControlUserPresence keychain helper (requires signed app)
        self._keychain_helper: Optional[Path] = None
        self._keychain_checked = False
        self._native_kc_works: Optional[bool] = None  # probed on first save

        # macOS: app-level Touch ID helper (works unsigned, fallback path)
        self._auth_helper: Optional[Path] = None
        self._auth_checked = False

        # Kick off compilation in the background so it's ready when needed
        if sys.platform == "darwin":
            threading.Thread(
                target=self._warmup_macos_helpers,
                daemon=True,
                name="keychain-helper-compile",
            ).start()

    # ------------------------------------------------------------------
    # Feature detection
    # ------------------------------------------------------------------

    def uses_native_secure_keychain(self) -> bool:
        """True when kSecAccessControlUserPresence is available.

        Requires: macOS, swiftc, AND a signed app with keychain entitlements.
        In development (unsigned binary): returns False, falls back to
        app-level Touch ID + standard keyring.
        """
        if sys.platform != "darwin":
            return False
        if self._native_kc_works is not None:
            return self._native_kc_works
        # Not yet probed — compilation may still be running
        return self._get_or_compile_keychain_helper() is not None and self._native_kc_works is not False

    def macos_touch_id_available(self) -> bool:
        """True when app-level Touch ID is available on macOS (compiled auth helper)."""
        if sys.platform != "darwin":
            return False
        return self._get_or_compile_auth_helper() is not None

    def biometric_available(self) -> bool:
        """True when an app-level biometric gate is useful for non-native-kc path.

        On macOS: Touch ID via auth helper (unless native_kc already handles it).
        On Windows: PowerShell Windows Hello gate.
        """
        if sys.platform == "darwin":
            return not self.uses_native_secure_keychain() and self.macos_touch_id_available()
        return os.name == "nt"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_profiles(self) -> List[ProfileMeta]:
        with self._lock:
            return self._read_index()

    def save_profile(self, profile: ServerProfile, pin: Optional[str] = None) -> None:
        """Encrypt and persist *profile*.

        On macOS with native keychain: *pin* is ignored — the OS enforces auth.
        On other platforms: *pin* adds an app-level verification layer.
        """
        with self._lock:
            self._profiles_dir.mkdir(parents=True, exist_ok=True)
            fernet_key = self._generate_fernet_key()
            payload_no_pin = json.dumps({"fernet_key": fernet_key.decode(), "has_pin": False})

            native_saved = False
            if self._get_or_compile_keychain_helper() is not None and self._native_kc_works is not False:
                # Try the native kSecAccessControlUserPresence path
                try:
                    self._native_keychain_set(profile.name, payload_no_pin)
                    self._native_kc_works = True
                    native_saved = True
                    profile.has_pin = False
                    self._encrypt_profile(profile, fernet_key)
                    self._update_index(profile.name, has_pin=False, add=True)
                except OSError as e:
                    if str(_ERR_MISSING_ENTITLEMENT) in str(e):
                        log.info(
                            "kSecAccessControlUserPresence unavailable (unsigned app) — "
                            "falling back to keyring + Touch ID gate"
                        )
                        self._native_kc_works = False
                    else:
                        raise

            if not native_saved:
                # Portable path: keyring + optional PIN hash
                if pin:
                    salt = os.urandom(16)
                    pin_hash = self._derive_pin_hash(pin, salt)
                    payload = json.dumps({
                        "fernet_key": fernet_key.decode(),
                        "has_pin": True,
                        "pin_hash": pin_hash.hex(),
                        "pin_salt": salt.hex(),
                    })
                    profile.has_pin = True
                else:
                    payload = payload_no_pin
                    profile.has_pin = False
                self._keyring_set(profile.name, payload)
                self._encrypt_profile(profile, fernet_key)
                self._update_index(profile.name, has_pin=bool(pin), add=True)

            log.info(
                "Saved server profile '%s' (native_kc=%s, pin=%s)",
                profile.name, native_saved, bool(pin),
            )

    def verify_pin(self, profile_name: str, pin: str) -> bool:
        """Return True if *pin* matches the stored hash (non-macOS path only)."""
        meta = self._keyring_get_meta(profile_name)
        if meta is None:
            return False
        if not meta.get("has_pin"):
            return True
        stored_hash = bytes.fromhex(meta["pin_hash"])
        salt = bytes.fromhex(meta["pin_salt"])
        return self._derive_pin_hash(pin, salt) == stored_hash

    def load_profile(self, profile_name: str, reason: str = "") -> Optional[ServerProfile]:
        """Decrypt and return a profile.

        On macOS with native secure keychain: triggers Touch ID automatically.
        Raises UserCancelledError if the user dismisses the biometric prompt.
        """
        with self._lock:
            if self._native_kc_works:
                r = reason or "Face Recognizer — szerverprofil betöltése"
                raw = self._native_keychain_get(profile_name, r)
                if raw is None:
                    return None
                meta = json.loads(raw)
            else:
                meta = self._keyring_get_meta(profile_name)
                if meta is None:
                    return None

            fernet_key = meta["fernet_key"].encode()
            return self._decrypt_profile(profile_name, fernet_key)

    def delete_profile(self, profile_name: str) -> None:
        with self._lock:
            if self.uses_native_secure_keychain():
                self._native_keychain_delete(profile_name)
            else:
                self._keyring_delete(profile_name)
            enc = self._profile_enc_path(profile_name)
            if enc.exists():
                enc.unlink()
            self._update_index(profile_name, has_pin=False, add=False)

    def rename_profile(self, old_name: str, new_name: str) -> bool:
        """Rename a profile. Returns False if *new_name* already exists."""
        with self._lock:
            if any(p.name == new_name for p in self._read_index()):
                return False

            # Read existing metadata / keychain entry
            if self.uses_native_secure_keychain():
                # get triggers Touch ID — skip for rename; just copy the raw bytes
                # by re-reading the encrypted payload via the helper
                raw = self._native_keychain_get(old_name, "Átnevezés")
                if raw is None:
                    return False
                meta: dict = json.loads(raw)
                payload = json.dumps(meta)
                fernet_key = meta["fernet_key"].encode()
                self._native_keychain_set(new_name, payload)
                self._native_keychain_delete(old_name)
            else:
                meta = self._keyring_get_meta(old_name)
                if meta is None:
                    return False
                fernet_key = meta["fernet_key"].encode()
                self._keyring_set(new_name, json.dumps(meta))
                self._keyring_delete(old_name)

            # Move and patch the encrypted settings file
            old_enc = self._profile_enc_path(old_name)
            new_enc = self._profile_enc_path(new_name)
            if old_enc.exists():
                old_enc.rename(new_enc)
                profile = self._decrypt_profile_from_file(new_enc, fernet_key)
                if profile:
                    profile.name = new_name
                    self._encrypt_profile_to(profile, fernet_key, new_enc)

            has_pin = meta.get("has_pin", False)
            self._update_index(old_name, has_pin=has_pin, add=False)
            self._update_index(new_name, has_pin=has_pin, add=True)
            return True

    # ------------------------------------------------------------------
    # Windows Hello (app-level gate, non-macOS)
    # ------------------------------------------------------------------

    def try_windows_hello(self, reason: str) -> bool:
        safe = reason.replace('"', "'")
        ps = (
            "try {"
            "$t=[Windows.Security.Credentials.UI.UserConsentVerifier,"
            "Windows.Security.Credentials.UI,ContentType=WindowsRuntime]"
            f'::RequestVerificationAsync("{safe}");'
            "$t.AsTask().Wait();"
            "if($t.Result -eq [Windows.Security.Credentials.UI.UserConsentVerificationResult]"
            "::Verified){exit 0} exit 1} catch {exit 2}"
        )
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                capture_output=True, timeout=60,
            )
            return r.returncode == 0
        except Exception as e:
            log.debug("Windows Hello error: %s", e)
            return False

    # ------------------------------------------------------------------
    # macOS warm-up (runs in background thread at startup)
    # ------------------------------------------------------------------

    def _warmup_macos_helpers(self) -> None:
        """Pre-compile both macOS helper binaries in the background."""
        self._get_or_compile_keychain_helper()
        self._get_or_compile_auth_helper()

    # ------------------------------------------------------------------
    # macOS app-level Touch ID (LAContext.evaluatePolicy — unsigned OK)
    # ------------------------------------------------------------------

    _SWIFT_AUTH_HELPER = """\
import LocalAuthentication
import Foundation

let reason = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "Authentication required"
let ctx = LAContext()
var err: NSError?
guard ctx.canEvaluatePolicy(.deviceOwnerAuthentication, error: &err) else { exit(2) }
let sem = DispatchSemaphore(value: 0)
var ok = false
ctx.evaluatePolicy(.deviceOwnerAuthentication, localizedReason: reason) { success, _ in
    ok = success
    sem.signal()
}
sem.wait()
exit(ok ? 0 : 1)
"""

    def try_macos_touch_id(self, reason: str) -> bool:
        """Trigger Touch ID / Face ID / password prompt via app-level LAContext."""
        helper = self._get_or_compile_auth_helper()
        if helper is None:
            return False
        try:
            r = subprocess.run([str(helper), reason], capture_output=True, timeout=120)
            return r.returncode == 0
        except Exception as e:
            log.debug("macOS Touch ID error: %s", e)
            return False

    def _get_or_compile_auth_helper(self) -> Optional[Path]:
        if self._auth_checked:
            return self._auth_helper
        with self._compile_lock:
            if self._auth_checked:
                return self._auth_helper
            self._auth_checked = True
            binary = _CACHE_DIR / "server_auth_helper"
            if binary.exists():
                self._auth_helper = binary
                return binary
            try:
                _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                src = _CACHE_DIR / "server_auth_helper.swift"
                src.write_text(self._SWIFT_AUTH_HELPER)
                r = subprocess.run(
                    ["swiftc", str(src), "-o", str(binary)],
                    capture_output=True, timeout=120,
                )
                if r.returncode == 0:
                    self._auth_helper = binary
                    log.info("Compiled auth helper: %s", binary)
                else:
                    log.debug("auth helper compilation failed: %s", r.stderr.decode(errors="replace"))
            except FileNotFoundError:
                log.debug("swiftc not found")
            except Exception as e:
                log.debug("auth helper compilation error: %s", e)
        return self._auth_helper

    # ------------------------------------------------------------------
    # macOS native Keychain helpers (kSecAccessControlUserPresence)
    # ------------------------------------------------------------------

    def _get_or_compile_keychain_helper(self) -> Optional[Path]:
        """Return path to compiled keychain_helper binary, compiling if needed.

        Thread-safe: only one compilation runs at a time; subsequent callers
        wait on the lock and then return the cached result.
        """
        if self._keychain_checked:
            return self._keychain_helper
        with self._compile_lock:
            # Re-check inside lock in case another thread compiled while we waited
            if self._keychain_checked:
                return self._keychain_helper
            self._keychain_checked = True
            binary = _CACHE_DIR / "keychain_helper"
            if binary.exists():
                self._keychain_helper = binary
                return binary
            try:
                _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                src = _CACHE_DIR / "keychain_helper.swift"
                src.write_text(_SWIFT_KEYCHAIN_HELPER)
                r = subprocess.run(
                    ["swiftc", str(src), "-o", str(binary)],
                    capture_output=True, timeout=120,
                )
                if r.returncode == 0:
                    self._keychain_helper = binary
                    log.info("Compiled keychain_helper: %s", binary)
                else:
                    log.warning(
                        "keychain_helper compilation failed:\n%s",
                        r.stderr.decode(errors="replace"),
                    )
            except FileNotFoundError:
                log.debug("swiftc not found — native Keychain helper unavailable")
            except Exception as e:
                log.debug("Swift compilation error: %s", e)
        return self._keychain_helper

    def _native_keychain_set(self, name: str, payload: str) -> None:
        """Store *payload* in macOS Keychain with kSecAccessControlUserPresence."""
        helper = self._keychain_helper
        if helper is None:
            raise RuntimeError("keychain_helper binary not available")
        r = subprocess.run(
            [str(helper), "set", _KEYRING_SERVICE, name],
            input=payload.encode("utf-8"),
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            raise OSError(
                f"keychain_helper set failed ({r.returncode}): "
                f"{r.stderr.decode(errors='replace')}"
            )

    def _native_keychain_get(self, name: str, reason: str) -> Optional[str]:
        """Read from macOS Keychain — triggers Touch ID / Face ID / password prompt.

        Returns None on failure or if the item does not exist.
        Raises _UserCancelledError if the user dismissed the biometric prompt.
        """
        helper = self._keychain_helper
        if helper is None:
            return None
        r = subprocess.run(
            [str(helper), "get", _KEYRING_SERVICE, name, reason],
            capture_output=True, timeout=120,  # allow time for biometric
        )
        if r.returncode == 0:
            return r.stdout.decode("utf-8")
        if r.returncode == 2:
            raise UserCancelledError("Biometric authentication cancelled")
        if r.returncode == 3:
            log.debug("No keychain item found for profile '%s'", name)
            return None
        log.warning(
            "keychain_helper get failed (%d): %s",
            r.returncode, r.stderr.decode(errors="replace"),
        )
        return None

    def _native_keychain_delete(self, name: str) -> None:
        helper = self._keychain_helper
        if helper is None:
            return
        subprocess.run(
            [str(helper), "delete", _KEYRING_SERVICE, name],
            capture_output=True, timeout=10,
        )

    # ------------------------------------------------------------------
    # Python keyring helpers (Windows / Linux / macOS fallback)
    # ------------------------------------------------------------------

    def _keyring_set(self, name: str, value: str) -> None:
        try:
            import keyring
            keyring.set_password(_KEYRING_SERVICE, name, value)
        except Exception as e:
            log.warning("Keyring set failed for '%s': %s", name, e)

    def _keyring_get_meta(self, name: str) -> Optional[dict]:
        try:
            import keyring
            raw = keyring.get_password(_KEYRING_SERVICE, name)
            if raw:
                return json.loads(raw)
        except Exception as e:
            log.debug("Keyring get failed for '%s': %s", name, e)
        return None

    def _keyring_delete(self, name: str) -> None:
        try:
            import keyring
            from keyring.errors import PasswordDeleteError
            keyring.delete_password(_KEYRING_SERVICE, name)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _generate_fernet_key(self) -> bytes:
        from cryptography.fernet import Fernet
        return Fernet.generate_key()

    @staticmethod
    def _derive_pin_hash(pin: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, 480_000)

    def _encrypt_profile(self, profile: ServerProfile, fernet_key: bytes) -> None:
        self._encrypt_profile_to(profile, fernet_key, self._profile_enc_path(profile.name))

    def _encrypt_profile_to(
        self, profile: ServerProfile, fernet_key: bytes, path: Path
    ) -> None:
        from cryptography.fernet import Fernet
        f = Fernet(fernet_key)
        data = json.dumps(profile.to_dict(), ensure_ascii=False).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f.encrypt(data))
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def _decrypt_profile(self, name: str, fernet_key: bytes) -> Optional[ServerProfile]:
        return self._decrypt_profile_from_file(self._profile_enc_path(name), fernet_key)

    def _decrypt_profile_from_file(
        self, path: Path, fernet_key: bytes
    ) -> Optional[ServerProfile]:
        if not path.exists():
            return None
        try:
            from cryptography.fernet import Fernet
            f = Fernet(fernet_key)
            data = f.decrypt(path.read_bytes())
            return ServerProfile.from_dict(json.loads(data.decode("utf-8")))
        except Exception as e:
            log.warning("Profile decrypt failed: %s", e)
            return None

    def _profile_enc_path(self, name: str) -> Path:
        safe = re.sub(r"[^\w\-]", "_", name)[:64]
        return self._profiles_dir / f"{safe}.enc"

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _read_index(self) -> List[ProfileMeta]:
        if not self._index_file.exists():
            return []
        try:
            data = json.loads(self._index_file.read_text(encoding="utf-8"))
            return [ProfileMeta(**p) for p in data.get("profiles", [])]
        except Exception as e:
            log.debug("Index read error: %s", e)
            return []

    def _update_index(self, name: str, *, has_pin: bool, add: bool) -> None:
        profiles = {p.name: p for p in self._read_index()}
        if add:
            profiles[name] = ProfileMeta(name=name, has_pin=has_pin)
        else:
            profiles.pop(name, None)
        self._index_file.write_text(
            json.dumps(
                {"profiles": [asdict(p) for p in profiles.values()]},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UserCancelledError(Exception):
    """Raised when the user dismisses the biometric / password prompt."""


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: Optional[ServerSettingsStore] = None
_store_lock = threading.Lock()


def get_server_settings_store() -> ServerSettingsStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ServerSettingsStore()
    return _store
