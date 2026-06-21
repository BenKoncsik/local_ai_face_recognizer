"""Interactive Leaflet map for picking a place coordinate.

Unlike the read-only :class:`~app.ui.widgets.place_map_widget.PlaceMapWidget`,
this widget is two-way: the user can drag the marker or click the map to set a
coordinate, and Python can move the marker / draw an accuracy circle from the
form. JS↔Python is bridged with ``QWebChannel``.

Signals:
    coordinate_picked(float, float)  emitted (lat, lon) when the user drags the
                                     marker or clicks the map.
"""

from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import QLabel, QStackedWidget, QWidget

from app.ui.i18n import t

try:
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineWidgets import QWebEngineView

    _WEBENGINE_OK = True
except ImportError:
    _WEBENGINE_OK = False


_MAP_HTML = """\
<!DOCTYPE html>
<html lang="hu">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{background:#1E1E2E}
    #map{width:100%;height:100vh}
    #offline-msg{
      display:none;position:absolute;top:8px;left:50%;transform:translateX(-50%);
      color:#F9E2AF;background:rgba(30,30,46,.92);padding:6px 14px;
      border-radius:6px;z-index:9999;font-size:.85rem;pointer-events:none;white-space:nowrap
    }
    .leaflet-container{background:#151515}
    #layer-btn{
      position:absolute;bottom:38px;right:10px;z-index:1000;
      background:#313244;color:#CDD6F4;border:1px solid #45475A;
      border-radius:6px;padding:5px 10px;cursor:pointer;font-size:.78rem;
      white-space:nowrap;line-height:1.4;box-shadow:0 1px 4px rgba(0,0,0,.4)
    }
    #layer-btn:hover{background:#45475A;border-color:#89B4FA}
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="offline-msg"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
  <script>
    var bridge=null, map=null, marker=null, circle=null;
    function ready(){
      map=L.map('map').setView([INIT_LAT,INIT_LON],INIT_ZOOM);
      var tileModern=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
        attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        errorTileUrl:'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
      });
      var tileHist=L.tileLayer('https://tiles.mapire.eu/mercator/europe-19century-thirdsurvey/{z}/{x}/{y}',{
        attribution:'&copy; <a href="https://mapire.eu">Mapire.eu</a> – Harmadik katonai felmérés (~1880)',
        maxZoom:15,
        errorTileUrl:'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
      });
      var isHist=false;
      tileModern.addTo(map);
      var errCnt=0;
      tileModern.on('tileerror',function(){
        if(++errCnt>=3){var el=document.getElementById('offline-msg');
          el.textContent=OFFLINE_MSG; el.style.display='block';}
      });
      map.on('click',function(e){ placeMarker(e.latlng.lat,e.latlng.lng,true); });

      var btn=document.createElement('button');
      btn.id='layer-btn';
      btn.textContent='Történelmi (1880)';
      document.getElementById('map').appendChild(btn);
      btn.addEventListener('click',function(){
        if(isHist){
          map.removeLayer(tileHist);
          tileModern.addTo(map);
          btn.textContent='Történelmi (1880)';
        } else {
          map.removeLayer(tileModern);
          tileHist.addTo(map);
          btn.textContent='Modern térkép';
        }
        isHist=!isHist;
      });
    }
    function placeMarker(lat,lon,notify){
      if(!marker){
        marker=L.marker([lat,lon],{draggable:true}).addTo(map);
        marker.on('dragend',function(){
          var p=marker.getLatLng(); if(bridge) bridge.markerMoved(p.lat,p.lng);
          if(circle) circle.setLatLng(p);
        });
      } else { marker.setLatLng([lat,lon]); }
      if(circle) circle.setLatLng([lat,lon]);
      if(notify && bridge) bridge.mapClicked(lat,lon);
    }
    // ---- Python → JS API ----
    function setMarker(lat,lon,zoom){
      placeMarker(lat,lon,false);
      if(zoom){ map.setView([lat,lon],zoom); } else { map.panTo([lat,lon]); }
    }
    function setAccuracyCircle(radius,color){
      if(!marker) return;
      var c=marker.getLatLng();
      if(!circle){
        circle=L.circle(c,{radius:radius,color:color,fillColor:color,
          weight:1,opacity:.7,fillOpacity:.12}).addTo(map);
      } else { circle.setRadius(radius); circle.setStyle({color:color,fillColor:color}); }
    }
    function clearMarker(){ if(marker){map.removeLayer(marker);marker=null;}
      if(circle){map.removeLayer(circle);circle=null;} }
    new QWebChannel(qt.webChannelTransport,function(ch){ bridge=ch.objects.bridge; ready(); });
  </script>
</body>
</html>
"""


class _MapBridge(QObject):
    """JS→Python bridge: the Leaflet page calls these slots."""

    moved = Signal(float, float)
    clicked = Signal(float, float)

    @Slot(float, float)
    def markerMoved(self, lat: float, lon: float) -> None:  # noqa: N802
        self.moved.emit(lat, lon)

    @Slot(float, float)
    def mapClicked(self, lat: float, lon: float) -> None:  # noqa: N802
        self.clicked.emit(lat, lon)


# Accuracy-circle colours per place type.
_CIRCLE_COLOR = {"exact": "#A6E3A1", "area": "#89B4FA", "region": "#F9E2AF"}


class PlaceMapPickerWidget(QStackedWidget):
    """Draggable/clickable Leaflet map. Page 0 = fallback label, page 1 = map."""

    coordinate_picked = Signal(float, float)

    _PAGE_FALLBACK = 0
    _PAGE_MAP = 1

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._view: Optional[QWidget] = None
        self._bridge: Optional[_MapBridge] = None
        self._channel = None
        self._ready = False
        self._pending_marker: Optional[tuple] = None
        self._pending_circle: Optional[tuple] = None

        self._fallback = QLabel()
        self._fallback.setAlignment(Qt.AlignCenter)
        self._fallback.setWordWrap(True)
        self._fallback.setStyleSheet(
            "color:#A6ADC8;background:#1E1E2E;padding:12px;font-size:0.9em;"
        )
        self.addWidget(self._fallback)

        self._placeholder = QLabel()
        self._placeholder.setStyleSheet("background:#1E1E2E;")
        self.addWidget(self._placeholder)

        if not _WEBENGINE_OK:
            self._fallback.setText(t("places_map_no_webengine"))
            self.setCurrentIndex(self._PAGE_FALLBACK)

    # ------------------------------------------------------------------

    def prepare(self, init_lat: float = 47.0, init_lon: float = 19.0, zoom: int = 7) -> None:
        """Create the web view and load the map. Call once after construction."""
        if not _WEBENGINE_OK:
            return
        if self._view is not None:
            return
        view = QWebEngineView()
        view.setMinimumHeight(300)
        self._bridge = _MapBridge()
        self._bridge.moved.connect(self._on_picked)
        self._bridge.clicked.connect(self._on_picked)
        self._channel = QWebChannel()
        self._channel.registerObject("bridge", self._bridge)
        view.page().setWebChannel(self._channel)
        view.loadFinished.connect(self._on_load_finished)

        self._view = view
        self.removeWidget(self._placeholder)
        self._placeholder.deleteLater()
        self.insertWidget(self._PAGE_MAP, view)
        self.setCurrentIndex(self._PAGE_MAP)

        html = (
            _MAP_HTML.replace("INIT_LAT", repr(float(init_lat)))
            .replace("INIT_LON", repr(float(init_lon)))
            .replace("INIT_ZOOM", str(int(zoom)))
            .replace("OFFLINE_MSG", json.dumps(t("places_map_offline")))
        )
        view.setHtml(html)

    def _on_load_finished(self, ok: bool) -> None:
        # The page calls ready() through the channel; give JS a moment then flush
        # any marker/circle requested before load completed.
        self._ready = bool(ok)
        if self._pending_marker is not None:
            self.set_marker(*self._pending_marker)
            self._pending_marker = None
        if self._pending_circle is not None:
            self.set_accuracy_circle(*self._pending_circle)
            self._pending_circle = None

    # ------------------------------------------------------------------
    # Python → JS
    # ------------------------------------------------------------------

    def set_marker(self, lat: float, lon: float, zoom: int = 15) -> None:
        if not _WEBENGINE_OK or self._view is None:
            return
        if not self._ready:
            self._pending_marker = (lat, lon, zoom)
            return
        self._run_js(f"setMarker({float(lat)!r},{float(lon)!r},{int(zoom)});")

    def set_accuracy_circle(self, radius_meters: float, place_type: str = "area") -> None:
        if not _WEBENGINE_OK or self._view is None:
            return
        color = _CIRCLE_COLOR.get(place_type, "#89B4FA")
        if not self._ready:
            self._pending_circle = (radius_meters, place_type)
            return
        self._run_js(f"setAccuracyCircle({float(radius_meters)!r},{json.dumps(color)});")

    def clear_marker(self) -> None:
        if _WEBENGINE_OK and self._view is not None and self._ready:
            self._run_js("clearMarker();")

    def _run_js(self, script: str) -> None:
        page = self._view.page()
        page.runJavaScript(script)

    # ------------------------------------------------------------------

    def _on_picked(self, lat: float, lon: float) -> None:
        self.coordinate_picked.emit(lat, lon)
