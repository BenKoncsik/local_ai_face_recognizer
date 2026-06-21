"""Verify the Gmail SMTP app-password baked into the exported server.

The exported Node server sends OTP login codes through Gmail using nodemailer's
``gmail`` service, i.e. ``smtp.gmail.com:465`` (implicit SSL). If the supplied
App Password is wrong, the running server fails silently at login time with a
``535 5.7.8 BadCredentials`` error and no mail goes out. These helpers let the
export dialog validate the exact same credentials *before* generating, so the
problem surfaces while it can still be fixed.
"""
from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional, Tuple

GMAIL_SMTP_HOST = "smtp.gmail.com"
GMAIL_SMTP_PORT = 465  # implicit SSL — matches nodemailer's `service: 'gmail'`


def _login_error(exc: Exception) -> str:
    """Human-readable Hungarian message for a failed SMTP login/send."""
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return (
            "A Gmail elutasította a belépést (helytelen cím vagy alkalmazás-jelszó).\n"
            "Ellenőrizd, hogy bekapcsoltad-e a kétlépcsős azonosítást, és hogy "
            "App Password-öt (16 karakter) adtál-e meg, nem a fiók jelszavát.\n\n"
            f"SMTP válasz: {exc}"
        )
    if isinstance(exc, (smtplib.SMTPConnectError, OSError, ssl.SSLError)):
        return (
            "Nem sikerült kapcsolódni a Gmail szerverhez "
            f"({GMAIL_SMTP_HOST}:{GMAIL_SMTP_PORT}). Ellenőrizd az internetkapcsolatot.\n\n"
            f"Hiba: {exc}"
        )
    return f"Email-küldési hiba: {exc}"


def check_credentials(
    email: str, app_password: str, *, timeout: float = 15.0
) -> Tuple[bool, str]:
    """Try to log in to Gmail SMTP with *email* / *app_password*.

    Returns ``(ok, message)``. ``ok`` is True when the login succeeds; otherwise
    ``message`` explains what went wrong (already localized).
    """
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(
            GMAIL_SMTP_HOST, GMAIL_SMTP_PORT, timeout=timeout, context=ctx
        ) as smtp:
            smtp.login(email.strip(), app_password.strip())
        return True, ""
    except Exception as exc:  # noqa: BLE001 — every failure is reported to the UI
        return False, _login_error(exc)


def send_test_email(
    email: str,
    app_password: str,
    *,
    to: Optional[str] = None,
    timeout: float = 15.0,
) -> Tuple[bool, str]:
    """Log in *and* actually send a short test message (default: to *email*).

    This is the strongest check — it proves the credentials can deliver mail,
    exactly like the OTP flow will. Returns ``(ok, message)``.
    """
    recipient = (to or email).strip()
    msg = EmailMessage()
    msg["From"] = email.strip()
    msg["To"] = recipient
    msg["Subject"] = "Galéria — teszt email"
    msg.set_content(
        "Ez egy teszt üzenet a Galéria exportból.\n\n"
        "Ha ezt megkaptad, az alkalmazás-jelszó működik, és a szerver "
        "képes lesz belépési kódokat küldeni."
    )
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(
            GMAIL_SMTP_HOST, GMAIL_SMTP_PORT, timeout=timeout, context=ctx
        ) as smtp:
            smtp.login(email.strip(), app_password.strip())
            smtp.send_message(msg)
        return True, recipient
    except Exception as exc:  # noqa: BLE001 — every failure is reported to the UI
        return False, _login_error(exc)
