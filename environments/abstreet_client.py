# abstreet_client.py
"""
Pure HTTP client for the A/B Street headless API.
No RL logic, no numpy arrays, no reward computation.
All methods return raw Python dicts/lists exactly as the API gives them.
"""
import requests
from typing import Optional, Dict, List, Any


class ABStreetClient:
    """
    Thin wrapper around the A/B Street headless REST API.
    One instance per headless server (per worker port).
    All methods raise on non-200 responses rather than silently returning None,
    so callers get explicit failures instead of silent bad state.
    """

    def __init__(self, host: str = "http://localhost", port: int = 1234, timeout: int = 30):
        self.base_url = f"{host}:{port}"
        self.timeout = timeout

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        resp = requests.get(
            f"{self.base_url}/{endpoint}",
            params=params,
            timeout=self.timeout
        )
        resp.raise_for_status()
        # Some endpoints return plain text (e.g. /sim/reset returns "Reset!")
        try:
            return resp.json()
        except Exception:
            return resp.text

    def _post(self, endpoint: str, payload: Any) -> Any:
        resp = requests.post(
            f"{self.base_url}/{endpoint}",
            json=payload,
            timeout=self.timeout
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return resp.text

    # ---- /sim ---------------------------------------------------------------

    def sim_load(self, scenario_path: str, modifiers: List = None, edits=None) -> str:
        return self._post("sim/load", {
            "scenario": scenario_path,
            "modifiers": modifiers or [],
            "edits": edits,
        })

    def sim_reset(self) -> str:
        return self._get("sim/reset")

    def sim_get_time(self) -> str:
        return self._get("sim/get-time")

    def sim_goto_time(self, time_str: str) -> str:
        """time_str must be HH:MM:SS format, e.g. '07:30:00'"""
        return self._get("sim/goto-time", params={"t": time_str})

    def sim_load_blank(self, map_path: str) -> str:
        return self._get("sim/load-blank", params={"map": map_path})

    # ---- /traffic-signals ---------------------------------------------------

    def get_all_signal_states(self) -> Dict:
        """
        Returns dict keyed by intersection ID.
        Each value contains 'waiting', 'stage', 'accepted' fields.
        """
        return self._get("traffic-signals/get-all-current-state")

    def get_signal(self, intersection_id: str) -> Dict:
        """Returns full ControlTrafficSignal for one intersection."""
        return self._get("traffic-signals/get", params={"id": intersection_id})

    def set_signal(self, signal_config: Dict) -> str:
        """Posts a modified ControlTrafficSignal back to the server."""
        return self._post("traffic-signals/set", signal_config)

    def get_signal_delays(
        self,
        intersection_id: str,
        t1: str,
        t2: str,
    ) -> Dict:
        """Delays experienced by agents through intersection_id from t1 to t2."""
        return self._get("traffic-signals/get-delays", params={
            "id": intersection_id,
            "t1": t1,
            "t2": t2,
        })

    def get_cumulative_throughput(self, intersection_id: str) -> Dict:
        """Agents passing through intersection since midnight, by direction."""
        return self._get(
            "traffic-signals/get-cumulative-thruput",
            params={"id": intersection_id}
        )

    # ---- /data --------------------------------------------------------------

    def get_finished_trips(self) -> List[Dict]:
        """
        Returns list of finished trips.
        Each dict: {id, duration (None if cancelled), mode, capped}
        """
        return self._get("data/get-finished-trips")

    def get_agent_positions(self) -> Dict:
        """Returns all active agents with vehicle_type, person ID, position."""
        return self._get("data/get-agent-positions")

    def get_road_throughput(self) -> List:
        """Returns (road, agent_type, hour, throughput) per road per hour."""
        return self._get("data/get-road-thruput")

    def get_blocked_by_graph(self) -> Dict:
        """Maps agent IDs to wait time and block reason."""
        return self._get("data/get-blocked-by-graph")

    def get_trip_time_lower_bound(self, trip_id: int) -> float:
        """Free-flow lower bound for one trip, in seconds."""
        return self._get("data/trip-time-lower-bound", params={"id": trip_id})

    def get_all_trip_time_lower_bounds(self) -> Dict[str, float]:
        """
        Batch equivalent of get_trip_time_lower_bound.
        Returns dict {trip_id_str: lower_bound_seconds}.
        API returns [[id, seconds], ...] — converted here so callers
        never see the raw list format.
        """
        raw = self._get("data/all-trip-time-lower-bounds")
        return {str(entry[0]): float(entry[1]) for entry in raw}

    # ---- /map ---------------------------------------------------------------

    def get_intersection_geometry(self, intersection_id: str) -> Dict:
        """GeoJSON with intersection polygon + connecting road features."""
        return self._get(
            "map/get-intersection-geometry",
            params={"id": intersection_id}
        )

    def get_all_geometry(self) -> Dict:
        """Full GeoJSON of every road and intersection. WGS84 coordinates."""
        return self._get("map/get-all-geometry")

    def get_nearest_road(
        self,
        lat: float,
        lon: float,
        threshold_meters: float = 100,
    ) -> Dict:
        """Snaps (lat, lon) to the nearest road centerline. Returns RoadID."""
        return self._get("map/get-nearest-road", params={
            "lat": lat,
            "lon": lon,
            "threshold_meters": threshold_meters,
        })

    def get_map_edits(self) -> Dict:
        return self._get("map/get-edits")

    def get_edit_road_command(self, road_id: str) -> Dict:
        return self._get("map/get-edit-road-command", params={"id": road_id})

    # ---- Helpers ------------------------------------------------------------

    def health_check(self) -> bool:
        """Returns True if the server is reachable and responding."""
        try:
            self.sim_get_time()
            return True
        except Exception:
            return False

    @staticmethod
    def seconds_to_hhmmss(total_seconds: float) -> str:
        """
        Converts raw seconds-after-midnight to HH:MM:SS string.
        Extracted from ABStreetIntersectionsEnv._seconds_to_hhmmss — belongs
        here since it's purely about the API's time format requirement.
        """
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
