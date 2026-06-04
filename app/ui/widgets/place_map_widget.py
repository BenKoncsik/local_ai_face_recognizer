"""Interactive Leaflet map widget for a single Place."""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QStackedWidget, QWidget

from app.ui.i18n import t

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _WEBENGINE_OK = True
except ImportError:
    _WEBENGINE_OK = False


# Static HTML shell — data is injected via <!-- DATA --> before setHtml()
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
      border-radius:6px;z-index:9999;font-size:.85rem;pointer-events:none;
      white-space:nowrap
    }
    .leaflet-container{background:#151515}
    .leaflet-popup-content-wrapper{background:#313244;color:#CDD6F4;border:none;
      box-shadow:0 2px 8px rgba(0,0,0,.5)}
    .leaflet-popup-tip{background:#313244}
    .leaflet-popup-close-button{color:#A6ADC8!important}
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="offline-msg" id="offline-msg"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    <!-- DATA -->
    (function(){
      var map=L.map('map').setView([LAT,LON],14);
      var tiles=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
        attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        errorTileUrl:'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
      }).addTo(map);

      var errCnt=0;
      tiles.on('tileerror',function(){
        if(++errCnt>=3){
          var el=document.getElementById('offline-msg');
          el.textContent=OFFLINE_MSG;
          el.style.display='block';
        }
      });

      var popup='<b>'+NAME+'</b><br>'+
        LAT.toFixed(6)+', '+LON.toFixed(6)+'<br>'+
        IMG_LABEL+': '+IMG_CNT+' &nbsp;|&nbsp; '+PER_LABEL+': '+PER_CNT;
      L.marker([LAT,LON]).addTo(map).bindPopup(popup).openPopup();

      NEARBY.forEach(function(pt){
        L.circleMarker([pt[0],pt[1]],{
          radius:5,fillColor:'#89B4FA',color:'#74C7EC',
          weight:1,opacity:.8,fillOpacity:.6
        }).addTo(map)
         .bindPopup(pt[0].toFixed(6)+', '+pt[1].toFixed(6));
      });
    })();
  </script>
</body>
</html>
"""


def _build_html(
    name: str,
    lat: float,
    lon: float,
    image_count: int,
    person_count: int,
    nearby: List[Tuple[float, float]],
) -> str:
    data = (
        f"var LAT={lat};\n"
        f"var LON={lon};\n"
        f"var NAME={json.dumps(name, ensure_ascii=False)};\n"
        f"var IMG_CNT={image_count};\n"
        f"var PER_CNT={person_count};\n"
        f"var NEARBY={json.dumps(nearby)};\n"
        f"var OFFLINE_MSG={json.dumps(t('places_map_offline'))};\n"
        f"var IMG_LABEL={json.dumps(t('places_image_count'))};\n"
        f"var PER_LABEL={json.dumps(t('places_person_count'))};\n"
    )
    return _MAP_HTML.replace("<!-- DATA -->", data)


class PlaceMapWidget(QStackedWidget):
    """QStackedWidget: page 0 = no-GPS label, page 1 = Leaflet QWebEngineView.

    QWebEngineView is created lazily on the first valid set_place() call.
    Creating it eagerly at construction time causes PySide6 SIGSEGV crashes:
    WebEngine's native child widgets don't receive Python wrappers until they
    are shown, so the main-window event filter crashes when it tries to look
    up those wrappers during early event dispatch.
    """

    _PAGE_NO_GPS = 0
    _PAGE_MAP = 1

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._view: Optional[QWidget] = None  # created lazily on first valid GPS

        self._no_gps_label = QLabel()
        self._no_gps_label.setAlignment(Qt.AlignCenter)
        self._no_gps_label.setWordWrap(True)
        self._no_gps_label.setStyleSheet(
            "color:#A6ADC8;background:#1E1E2E;padding:12px;font-size:0.9em;"
        )
        self.addWidget(self._no_gps_label)

        # Placeholder for page 1 until the real view is created
        self._map_placeholder = QLabel()
        self._map_placeholder.setStyleSheet("background:#1E1E2E;")
        self.addWidget(self._map_placeholder)

        self.setCurrentIndex(self._PAGE_NO_GPS)
        self._no_gps_label.setText(t("places_map_no_gps"))

    def _ensure_view(self) -> QWidget:
        """Create and install the real map view the first time GPS data arrives."""
        if self._view is not None:
            return self._view

        if _WEBENGINE_OK:
            view: QWidget = QWebEngineView()
            view.setMinimumHeight(200)
        else:
            lbl = QLabel(t("places_map_no_webengine"))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#A6ADC8;background:#1E1E2E;padding:12px;")
            view = lbl

        self._view = view
        # Swap out the placeholder at page index 1
        self.removeWidget(self._map_placeholder)
        self._map_placeholder.deleteLater()
        self.insertWidget(self._PAGE_MAP, view)
        return view

    def set_place(
        self,
        name: str,
        lat: Optional[float],
        lon: Optional[float],
        image_count: int,
        person_count: int,
        nearby_points: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        is_valid = (
            lat is not None
            and lon is not None
            and not (abs(lat) < 1e-6 and abs(lon) < 1e-6)
        )
        if not is_valid:
            self._no_gps_label.setText(t("places_map_no_gps"))
            self.setCurrentIndex(self._PAGE_NO_GPS)
            return

        view = self._ensure_view()
        if not _WEBENGINE_OK or not isinstance(view, QWebEngineView):
            self.setCurrentIndex(self._PAGE_MAP)
            return

        html = _build_html(
            name=name,
            lat=lat,
            lon=lon,
            image_count=image_count,
            person_count=person_count,
            nearby=nearby_points or [],
        )
        view.setHtml(html)
        self.setCurrentIndex(self._PAGE_MAP)

    def clear(self) -> None:
        self._no_gps_label.setText(t("places_map_no_gps"))
        self.setCurrentIndex(self._PAGE_NO_GPS)
        if _WEBENGINE_OK and isinstance(self._view, QWebEngineView):
            self._view.setHtml("")
