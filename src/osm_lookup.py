"""OSM infrastructure lookup for the 9 binary road features used in the model."""
import sqlite3
import warnings
from pathlib import Path

INFRA_FEATURES = [
    'Amenity', 'Crossing', 'Give_Way', 'Junction',
    'No_Exit', 'Railway', 'Station', 'Stop', 'Traffic_Signal',
]

DEFAULT_INFRA = {k: 0 for k in INFRA_FEATURES}

# OSM tags that map to the 9 model features
_QUERY_TAGS = {
    'highway'        : ['traffic_signals', 'crossing', 'stop', 'give_way', 'motorway_junction'],
    'junction'       : True,
    'railway'        : True,
    'amenity'        : True,
    'public_transport': ['station'],
    'noexit'         : ['yes'],
}

_BIN_SIZE   = 0.005   # ~500 m per bin
_QUERY_DIST = 300     # metres radius for OSM query


def _to_bin(lat: float, lng: float) -> tuple:
    return round(lat / _BIN_SIZE), round(lng / _BIN_SIZE)


def _tags_to_features(gdf) -> dict:
    feats = {k: 0 for k in INFRA_FEATURES}
    for _, row in gdf.iterrows():
        hw = str(row.get('highway', '') or '').lower()
        jn = row.get('junction')
        rw = str(row.get('railway', '') or '').lower()
        am = row.get('amenity')
        pt = str(row.get('public_transport', '') or '').lower()
        ne = str(row.get('noexit', '') or '').lower()

        if 'traffic_signals'   in hw: feats['Traffic_Signal'] = 1
        if 'crossing'          in hw: feats['Crossing']       = 1
        if 'stop'              in hw: feats['Stop']           = 1
        if 'give_way'          in hw: feats['Give_Way']       = 1
        if 'motorway_junction' in hw: feats['Junction']       = 1
        if jn and str(jn).lower() not in ('nan', ''):
                                      feats['Junction']       = 1
        if rw and rw not in ('nan', ''):
            feats['Station' if 'station' in rw else 'Railway'] = 1
        if 'station' in pt:           feats['Station']        = 1
        if am and str(am).lower() not in ('nan', ''):
                                      feats['Amenity']        = 1
        if ne == 'yes':               feats['No_Exit']        = 1
    return feats


def _init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS infra (
            lat_bin INTEGER, lng_bin INTEGER,
            Amenity INTEGER, Crossing INTEGER, Give_Way INTEGER,
            Junction INTEGER, No_Exit INTEGER, Railway INTEGER,
            Station INTEGER, Stop INTEGER, Traffic_Signal INTEGER,
            PRIMARY KEY (lat_bin, lng_bin)
        )
    """)
    conn.commit()
    return conn


def _read_cache(conn: sqlite3.Connection, lat_bin: int, lng_bin: int):
    cur = conn.execute(
        "SELECT Amenity,Crossing,Give_Way,Junction,No_Exit,Railway,"
        "Station,Stop,Traffic_Signal "
        "FROM infra WHERE lat_bin=? AND lng_bin=?",
        (lat_bin, lng_bin),
    )
    row = cur.fetchone()
    return dict(zip(INFRA_FEATURES, row)) if row is not None else None


def _write_cache(conn: sqlite3.Connection, lat_bin: int, lng_bin: int, feats: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO infra VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (lat_bin, lng_bin,
         feats['Amenity'],  feats['Crossing'],  feats['Give_Way'],
         feats['Junction'], feats['No_Exit'],   feats['Railway'],
         feats['Station'],  feats['Stop'],      feats['Traffic_Signal']),
    )
    conn.commit()


def get_infra_features(lat: float, lng: float,
                       db_path: str = 'models/infra_lookup.db') -> dict:
    """Return the 9 binary infra features for a GPS coordinate.

    Checks SQLite cache first; on cache miss, queries OSM live and caches the
    result. Returns all-zero defaults if OSM is unreachable.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    lat_bin, lng_bin = _to_bin(lat, lng)

    conn = _init_db(db_path)
    cached = _read_cache(conn, lat_bin, lng_bin)
    if cached is not None:
        conn.close()
        print(f"  Infra (cache): {cached}")
        return cached

    # Live OSM query
    try:
        import osmnx as ox
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gdf = ox.features_from_point((lat, lng), tags=_QUERY_TAGS, dist=_QUERY_DIST)
        feats = _tags_to_features(gdf)
        print(f"  Infra (OSM) : {feats}")
    except Exception as e:
        print(f"  Infra (default, OSM error: {e})")
        feats = DEFAULT_INFRA.copy()

    _write_cache(conn, lat_bin, lng_bin, feats)
    conn.close()
    return feats
