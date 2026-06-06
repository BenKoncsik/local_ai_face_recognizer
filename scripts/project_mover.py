#!/usr/bin/env python3
"""
project_mover.py — interaktív kis terminálos alkalmazás GitHub Projects (v2)
elemek áthelyezéséhez két board között, állapot (Status) alapján.

Folyamat (kapcsolók nélkül, menüvezérelt):
  1. Bejelentkezés (gh auth) — ha kell, elindítja a bejelentkezést.
  2. Projektek listázása, FORRÁS kiválasztása.
  3. Projektek listázása, CÉL kiválasztása.
  4. Áthelyezendő állapotok kiválasztása (a forrásban előfordulók közül).
  5. Áthelyezés végrehajtása.
  6. Összegzés: melyik állapotba hány jegy került.

"Áthelyezés":
  - valódi Issue / PR  -> ugyanazt a kártyát hozzáadja a cél-projekthez,
                          majd a forrásból eltávolítja (átmozgatás).
  - draft elem         -> új draft kártya a célban (cím + leírás), majd a
                          forrás draft törlése.
  A cél-projektben beállítja ugyanazt a Status értéket.

Követelmény: gh (GitHub CLI), és a tokenen 'project' scope (a script felajánlja).
Futtatás:  python3 scripts/project_mover.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

# ---- terminál színek -------------------------------------------------------
class C:
    R = "\033[31m"; G = "\033[32m"; Y = "\033[33m"; B = "\033[36m"
    BOLD = "\033[1m"; DIM = "\033[2m"; X = "\033[0m"

def cprint(s: str, color: str = "") -> None:
    print(f"{color}{s}{C.X}")

def die(msg: str) -> None:
    cprint(f"✗ {msg}", C.R); sys.exit(1)

def ok(msg: str) -> None:
    cprint(f"✓ {msg}", C.G)

def info(msg: str) -> None:
    cprint(f"• {msg}", C.B)

def warn(msg: str) -> None:
    cprint(f"! {msg}", C.Y)


# ---- gh segédek ------------------------------------------------------------
def gh(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gh", *args],
        text=True,
        capture_output=capture,
        check=check,
    )

def gh_json(*args: str) -> Any:
    res = gh(*args)
    return json.loads(res.stdout) if res.stdout.strip() else None


def ensure_auth() -> None:
    """Bejelentkezés + 'project' scope biztosítása."""
    try:
        gh("auth", "status", check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        warn("Nem vagy bejelentkezve. Indítom a bejelentkezést…")
        try:
            subprocess.run(["gh", "auth", "login"], check=True)
        except subprocess.CalledProcessError:
            die("A bejelentkezés nem sikerült.")
    # project scope ellenőrzése egy próbahívással
    probe = gh("project", "list", "--owner", "@me", "--limit", "1", check=False)
    if probe.returncode != 0:
        warn("Hiányzik a 'project' jogosultság a tokenből. Hozzáadom…")
        try:
            subprocess.run(["gh", "auth", "refresh", "-s", "project"], check=True)
        except subprocess.CalledProcessError:
            die("Nem sikerült hozzáadni a 'project' jogosultságot. Futtasd kézzel: gh auth refresh -s project")
    ok("Bejelentkezve.")


# ---- bemeneti segédek ------------------------------------------------------
def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{C.BOLD}{prompt}{suffix}: {C.X}").strip()
    return val or default

def choose_one(items: list[dict], render, title: str) -> dict:
    print()
    cprint(title, C.BOLD)
    for i, it in enumerate(items, 1):
        print(f"  {C.B}[{i}]{C.X} {render(it)}")
    while True:
        sel = input(f"{C.BOLD}Választás (1-{len(items)}): {C.X}").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(items):
            return items[int(sel) - 1]
        warn("Érvénytelen választás.")

def choose_many(options: list[str], counts: dict[str, int], title: str) -> list[str]:
    """Több állapot kiválasztása vesszővel/szóközzel, vagy 'all'."""
    print()
    cprint(title, C.BOLD)
    for i, opt in enumerate(options, 1):
        print(f"  {C.B}[{i}]{C.X} {opt}  {C.DIM}({counts.get(opt, 0)} jegy){C.X}")
    print(f"  {C.DIM}Pl.: 1,3,4  vagy  'all' az összeshez{C.X}")
    while True:
        raw = input(f"{C.BOLD}Áthelyezendő állapotok: {C.X}").strip().lower()
        if raw in ("all", "*"):
            return list(options)
        parts = [p for p in raw.replace(",", " ").split() if p]
        idxs = [int(p) for p in parts if p.isdigit()]
        if idxs and all(1 <= n <= len(options) for n in idxs):
            return [options[n - 1] for n in sorted(set(idxs))]
        warn("Érvénytelen választás.")


# ---- projekt műveletek -----------------------------------------------------
def pick_owner(label: str, default: str = "@me") -> str:
    print()
    cprint(label, C.BOLD)
    return ask("  Owner login (user vagy szervezet), Enter = saját user", default)

def list_projects(owner: str) -> list[dict]:
    data = gh_json("project", "list", "--owner", owner, "--format", "json", "--limit", "100")
    projects = (data or {}).get("projects", [])
    if not projects:
        die(f"Nincs projekt ehhez az ownerhez: {owner}")
    return projects

def status_field(owner: str, number: int, field_name: str = "Status") -> dict:
    data = gh_json("project", "field-list", str(number), "--owner", owner, "--format", "json")
    for f in (data or {}).get("fields", []):
        if f.get("name") == field_name:
            if not f.get("options"):
                die(f"A(z) '{field_name}' mező nem single-select a projektben (#{number}).")
            return f
    die(f"Nincs '{field_name}' mező a projektben (#{number}).")

def field_key(name: str) -> str:
    """A gh item-list a mezőket camelCase kulccsal adja vissza (Status -> status)."""
    words = name.split()
    if not words:
        return name
    out = words[0][:1].lower() + words[0][1:]
    for w in words[1:]:
        out += w[:1].upper() + w[1:]
    return out

def fetch_items(owner: str, number: int) -> list[dict]:
    data = gh_json("project", "item-list", str(number), "--owner", owner,
                   "--format", "json", "--limit", "2000")
    return (data or {}).get("items", [])


def main() -> None:
    cprint("\n  GitHub Projects – jegy áthelyező\n", C.BOLD + C.B)
    ensure_auth()

    # --- forrás projekt ---
    src_owner = pick_owner("FORRÁS projekt owner:")
    src_projects = list_projects(src_owner)
    source = choose_one(src_projects,
                        lambda p: f"#{p['number']}  {p['title']}",
                        f"FORRÁS projekt (owner: {src_owner}):")

    # --- cél projekt ---
    tgt_owner = pick_owner("CÉL projekt owner:", default=src_owner)
    tgt_projects = list_projects(tgt_owner)
    target = choose_one(tgt_projects,
                        lambda p: f"#{p['number']}  {p['title']}",
                        f"CÉL projekt (owner: {tgt_owner}):")

    if src_owner == tgt_owner and source["number"] == target["number"]:
        die("A forrás és a cél projekt ugyanaz.")

    # --- cél Status mező ---
    tgt_view = gh_json("project", "view", str(target["number"]), "--owner", tgt_owner, "--format", "json")
    target_project_id = tgt_view["id"]
    tgt_status = status_field(tgt_owner, target["number"])
    status_opt_id = {o["name"]: o["id"] for o in tgt_status["options"]}
    status_field_id = tgt_status["id"]

    # --- forrás elemek + állapot szerinti csoportosítás ---
    info("Forrás elemek lekérése…")
    items = fetch_items(src_owner, source["number"])
    skey = field_key("Status")

    counts: dict[str, int] = {}
    for it in items:
        st = it.get(skey)
        if st:
            counts[st] = counts.get(st, 0) + 1
    if not counts:
        die("A forrás projektben nincs Status értékkel rendelkező elem.")

    statuses = sorted(counts.keys())
    chosen = choose_many(statuses, counts, "Mely állapotokat helyezzük át?")

    selected = [it for it in items if it.get(skey) in chosen]
    print()
    info(f"Forrás:  #{source['number']}  {source['title']}  ({src_owner})")
    info(f"Cél:     #{target['number']}  {target['title']}  ({tgt_owner})")
    info(f"Áthelyezendő elemek: {len(selected)}")
    for st in chosen:
        n = sum(1 for it in selected if it.get(skey) == st)
        missing = "" if st in status_opt_id else f"  {C.Y}(a célban nincs ilyen állapot – Status nem lesz beállítva){C.X}"
        print(f"   {C.DIM}-{C.X} {st}: {n}{missing}")

    if input(f"\n{C.BOLD}Folytatod? [y/N] {C.X}").strip().lower() not in ("y", "i"):
        warn("Megszakítva."); sys.exit(1)

    def set_status(item_id: str, status_name: str) -> None:
        gh("project", "item-edit", "--id", item_id,
           "--project-id", target_project_id,
           "--field-id", status_field_id,
           "--single-select-option-id", status_opt_id[status_name])

    # --- áthelyezés ---
    moved: dict[str, int] = {st: 0 for st in chosen}
    failed = 0
    # (cél item id, kívánt státusz, cím) — az ellenőrző körhöz
    placed: list[tuple[str, str, str]] = []
    print()
    for it in selected:
        st = it.get(skey)
        content = it.get("content") or {}
        ctype = content.get("type", "DraftIssue")
        title = content.get("title") or it.get("title") or "(cím nélkül)"
        url = content.get("url", "")
        src_item_id = it.get("id")

        try:
            if ctype == "DraftIssue" or not url:
                body = content.get("body", "")
                new = gh_json("project", "item-create", str(target["number"]),
                              "--owner", tgt_owner, "--title", title,
                              "--body", body, "--format", "json")
            else:
                new = gh_json("project", "item-add", str(target["number"]),
                              "--owner", tgt_owner, "--url", url, "--format", "json")
            new_id = new["id"]

            # Status beállítása a célban
            if st in status_opt_id:
                set_status(new_id, st)
                placed.append((new_id, st, title))

            # forrásból eltávolítás (átmozgatás)
            if src_item_id:
                gh("project", "item-delete", str(source["number"]),
                   "--owner", src_owner, "--id", src_item_id, check=False)

            moved[st] += 1
            ok(f"áthelyezve [{st}]: {title}")
        except subprocess.CalledProcessError as e:
            failed += 1
            warn(f"SIKERTELEN: {title}  ({(e.stderr or '').strip()[:120]})")

    # --- ellenőrző / korrekciós kör ---
    # A cél-projekt beépített automatizmusai (pl. "Item closed -> Done") a
    # hozzáadás után felülírhatják a státuszt. Visszaolvassuk a tényleges
    # értékeket, és az eltéréseket újra beállítjuk (max 3 próba).
    if placed:
        print()
        info("Státuszok ellenőrzése (automatizmusok felülírhatták)…")
        for attempt in range(3):
            time.sleep(2)
            actual = {i["id"]: i.get(skey) for i in fetch_items(tgt_owner, target["number"])}
            mismatched = [(iid, st, title) for (iid, st, title) in placed
                          if actual.get(iid) != st]
            if not mismatched:
                ok("Minden státusz rendben.")
                break
            for iid, st, title in mismatched:
                try:
                    set_status(iid, st)
                    warn(f"korrigálva [{st}]: {title}")
                except subprocess.CalledProcessError:
                    pass
        else:
            still = [t for (iid, st, t) in placed
                     if actual.get(iid) != st]
            if still:
                warn(f"Nem sikerült korrigálni {len(still)} jegy státuszát "
                     f"(lehet, hogy az automatizmus újra felülírja): {', '.join(still)}")

    # --- összegzés ---
    print()
    cprint("  Összegzés", C.BOLD + C.G)
    total = 0
    for st in chosen:
        cprint(f"   {st}: {moved[st]} jegy", C.G)
        total += moved[st]
    cprint(f"   Összesen áthelyezve: {total}", C.BOLD + C.G)
    if failed:
        warn(f"   Sikertelen: {failed}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        warn("Megszakítva.")
        sys.exit(130)
