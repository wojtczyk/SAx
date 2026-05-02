# SAx Story

**SAx** means **Search And x**.

The variable **x** changes with the mission:

- **R**: Rescue
- **T**: Track
- **C**: Classify
- **D**: Detect or Disable
- **M**: Map or Monitor
- **I**: Identify
- **P**: Protect
- **L**: Locate

For the current hackathon demo, **SAx = Search And Detect**.

The prototype supports mission profiles that change which detections are treated as high priority and how the SITREP is phrased:

- **SA-R**: Search And Rescue
- **SA-D**: Search And Detect
- **SA-T**: Search And Track
- **SA-M**: Search And Map
- **SA-P**: Search And Protect

## Mission

SAx is an edge mission assistant for disconnected environments. When floods, earthquakes, hurricanes, or contested conditions knock out power, landlines, and cell service, SAx helps small teams search, detect, track, locate, and report what matters using local drones, sensors, and speech.

## Why It Matters

Disaster response and national security teams often work in places where normal infrastructure is degraded or unavailable. They may have limited bandwidth, limited power, too many live feeds, and too few people watching them.

SAx turns raw observations into structured mission intelligence:

- Drone or camera video provides local eyes on the scene.
- YOLO detects people, vehicles, supplies, obstacles, and other objects of interest.
- Speech-to-text captures hands-free operator notes.
- A local event timeline keeps detections and notes organized.
- A SITREP summarizes recent activity into a concise situation report.
- A future LoRa mode can relay compact alerts when normal networks are down.

## Demo Narrative

A hurricane has knocked out cell service and power across part of a city. A response team launches a low-cost drone from a staging area to inspect flooded streets, blocked roads, rooftops, and damaged buildings.

SAx runs locally on a Mac edge station. It analyzes the video feed, identifies relevant objects, captures responder voice notes, and generates a short SITREP from the recent detections and notes.

The operator does not need cloud access to understand what is happening. The system helps convert noisy sensor data into actionable updates that can be shared with a team, relayed over low-bandwidth links, or used to decide where to search next.

## Dual Use

The same core workflow can support:

- Search and rescue after floods, earthquakes, hurricanes, and wildfires
- Route reconnaissance when roads or bridges may be blocked
- Perimeter monitoring around temporary shelters, bases, or staging areas
- Force protection in austere or contested environments
- Low-bandwidth reporting when normal communications are unavailable

The project is intentionally framed around finding, understanding, and protecting under degraded conditions.

## Current Tagline

**SAx turns low-cost drones, sensors, and operator speech into structured mission intelligence at the edge.**
