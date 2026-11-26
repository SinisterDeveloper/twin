import { feature } from "topojson-client";
import worldCountries from "world-atlas/countries-50m.json";
import indiaStatesRaw from "../assets/india_states.json";
import { LAT_MAX, LAT_MIN, LON_MAX, LON_MIN } from "./contract";

const SVG_W = 360;
const SVG_H = 320;

function lonToX(lon: number): number {
  return ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * SVG_W;
}

function latToY(lat: number): number {
  return ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * SVG_H;
}

type Position = number[];
type PolygonCoords = Position[][];
type MultiPolygonCoords = Position[][][];

interface GeoFeature {
  id?: string | number;
  properties?: Record<string, unknown>;
  geometry: {
    type: string;
    coordinates: unknown;
  } | null;
}

function ringToPath(ring: Position[]): string {
  if (ring.length < 2) return "";
  let d = "";
  for (let i = 0; i < ring.length; i++) {
    const [lon, lat] = ring[i];
    const x = lonToX(lon).toFixed(1);
    const y = latToY(lat).toFixed(1);
    d += i === 0 ? `M${x},${y}` : `L${x},${y}`;
  }
  return `${d}Z`;
}

function ringsIntersectDomain(rings: Position[][]): boolean {
  let minLon = Infinity;
  let maxLon = -Infinity;
  let minLat = Infinity;
  let maxLat = -Infinity;
  for (const ring of rings) {
    for (const [lon, lat] of ring) {
      if (lon < minLon) minLon = lon;
      if (lon > maxLon) maxLon = lon;
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
    }
  }
  return maxLon >= LON_MIN - 2 && minLon <= LON_MAX + 2 && maxLat >= LAT_MIN - 2 && minLat <= LAT_MAX + 2;
}

function geometryToPath(geom: GeoFeature["geometry"]): string {
  if (!geom) return "";
  if (geom.type === "Polygon") {
    const rings = geom.coordinates as PolygonCoords;
    if (!ringsIntersectDomain(rings)) return "";
    return rings.map(ringToPath).join("");
  }
  if (geom.type === "MultiPolygon") {
    const polys = geom.coordinates as MultiPolygonCoords;
    const parts: string[] = [];
    for (const poly of polys) {
      if (!ringsIntersectDomain(poly)) continue;
      parts.push(poly.map(ringToPath).join(""));
    }
    return parts.join("");
  }
  return "";
}

export interface NamedGeoPath {
  name: string;
  d: string;
}

// 1. Indian states & UTs (36 polygons from bundled GeoJSON)
export const INDIA_STATE_PATHS: NamedGeoPath[] = (
  indiaStatesRaw.features as GeoFeature[]
)
  .map((f) => ({
    name: String(f.properties?.name ?? "State"),
    d: geometryToPath(f.geometry),
  }))
  .filter((item) => item.d.length > 0);

// 2. Surrounding South Asia coastlines & countries from world-atlas 50m (excluding India 356)
const topoArg = worldCountries as unknown as Parameters<typeof feature>[0];
const objArg = (worldCountries as { objects: { countries: unknown } }).objects
  .countries as Parameters<typeof feature>[1];
const worldCollection = feature(topoArg, objArg) as unknown as {
  features: GeoFeature[];
};

export const NEIGHBOR_COUNTRY_PATHS: NamedGeoPath[] = worldCollection.features
  .filter((f) => String(f.id ?? "") !== "356")
  .map((f) => ({
    name: String(f.properties?.name ?? "Country"),
    d: geometryToPath(f.geometry),
  }))
  .filter((item) => item.d.length > 0);

// 3. India national outline only (used when internal State Borders are hidden)
export const INDIA_OUTLINE_PATHS: NamedGeoPath[] = worldCollection.features
  .filter((f) => String(f.id ?? "") === "356")
  .map((f) => ({
    name: String(f.properties?.name ?? "India"),
    d: geometryToPath(f.geometry),
  }))
  .filter((item) => item.d.length > 0);
