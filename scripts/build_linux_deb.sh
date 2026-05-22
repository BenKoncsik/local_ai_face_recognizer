#!/usr/bin/env bash

set -euo pipefail

VERSION="${1:?Usage: build_linux_deb.sh <version> [dist_dir] [release_dir]}"
DIST_DIR="${2:-dist/Face-Local}"
RELEASE_DIR="${3:-release}"

APP_SLUG="face-local"
APP_NAME="Face-Local"
ARCH="$(dpkg --print-architecture)"
case "$ARCH" in
    amd64) ASSET_ARCH="x64" ;;
    *) ASSET_ARCH="$ARCH" ;;
esac
PKG_DIR="$RELEASE_DIR/${APP_SLUG}_${VERSION}_${ARCH}"
INSTALL_DIR="$PKG_DIR/opt/$APP_SLUG"

rm -rf "$PKG_DIR"
mkdir -p \
    "$PKG_DIR/DEBIAN" \
    "$PKG_DIR/usr/bin" \
    "$PKG_DIR/usr/share/applications" \
    "$PKG_DIR/usr/share/icons/hicolor/512x512/apps" \
    "$INSTALL_DIR"

cp -R "$DIST_DIR"/. "$INSTALL_DIR"/

cat > "$PKG_DIR/DEBIAN/control" <<EOF
Package: $APP_SLUG
Version: $VERSION
Section: utils
Priority: optional
Architecture: $ARCH
Maintainer: Face-Local
Depends: libgl1, libglib2.0-0, libxkbcommon-x11-0, libxcb-cursor0
Description: Offline face grouping and person labeling desktop app
EOF

cat > "$PKG_DIR/usr/bin/$APP_SLUG" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /opt/face-local
exec ./Face-Local "$@"
EOF
chmod 755 "$PKG_DIR/usr/bin/$APP_SLUG"

install -m 644 "scripts/linux/face-local.desktop" "$PKG_DIR/usr/share/applications/face-local.desktop"
install -m 644 "assets/icons/app-icon-512.png" \
    "$PKG_DIR/usr/share/icons/hicolor/512x512/apps/face-local.png"

dpkg-deb --build --root-owner-group "$PKG_DIR"
mv "$PKG_DIR.deb" "$RELEASE_DIR/${APP_NAME}-linux-${ASSET_ARCH}-installer-${VERSION}.deb"

tar -C dist -czf "$RELEASE_DIR/${APP_NAME}-linux-${ASSET_ARCH}-portable-${VERSION}.tar.gz" "$APP_NAME"
