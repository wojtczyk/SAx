from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MissionProfile:
    name: str
    code: str
    description: str
    priority_labels: frozenset[str]
    detection_term: str
    assessment_focus: str
    recommendation: str

    def is_priority(self, label: str) -> bool:
        return label.lower() in self.priority_labels


MISSION_PROFILES: dict[str, MissionProfile] = {
    "Search And Rescue": MissionProfile(
        name="Search And Rescue",
        code="SA-R",
        description="Find people, vehicles, boats, and supplies after a disaster.",
        priority_labels=frozenset(
            {
                "person",
                "car",
                "truck",
                "bus",
                "boat",
                "backpack",
                "suitcase",
                "cell phone",
            }
        ),
        detection_term="rescue-relevant object",
        assessment_focus="survivors, stranded vehicles, access routes, and useful supplies",
        recommendation="Prioritize likely survivors, capture another frame, and add location context for responders.",
    ),
    "Search And Detect": MissionProfile(
        name="Search And Detect",
        code="SA-D",
        description="Detect activity and objects of interest in an area.",
        priority_labels=frozenset(
            {
                "person",
                "car",
                "truck",
                "bus",
                "motorcycle",
                "bicycle",
                "backpack",
                "suitcase",
            }
        ),
        detection_term="object of interest",
        assessment_focus="people, vehicles, carried objects, and changes in the scene",
        recommendation="Continue observation, capture a confirming frame, and add an operator note for context.",
    ),
    "Search And Track": MissionProfile(
        name="Search And Track",
        code="SA-T",
        description="Track movement of people and vehicles across repeated observations.",
        priority_labels=frozenset(
            {
                "person",
                "car",
                "truck",
                "bus",
                "motorcycle",
                "bicycle",
                "boat",
            }
        ),
        detection_term="trackable object",
        assessment_focus="movement, repeat sightings, and direction of travel",
        recommendation="Keep the sensor fixed on the area and collect repeated frames to establish movement.",
    ),
    "Search And Map": MissionProfile(
        name="Search And Map",
        code="SA-M",
        description="Map roads, vehicles, obstacles, and access constraints.",
        priority_labels=frozenset(
            {
                "person",
                "car",
                "truck",
                "bus",
                "boat",
                "traffic light",
                "stop sign",
                "bench",
                "fire hydrant",
            }
        ),
        detection_term="map-relevant object",
        assessment_focus="routes, landmarks, blocked access, and staging areas",
        recommendation="Add a location note and capture overlapping views to improve local map context.",
    ),
    "Search And Protect": MissionProfile(
        name="Search And Protect",
        code="SA-P",
        description="Monitor areas around shelters, staging points, or temporary perimeters.",
        priority_labels=frozenset(
            {
                "person",
                "car",
                "truck",
                "bus",
                "motorcycle",
                "bicycle",
                "backpack",
                "suitcase",
            }
        ),
        detection_term="protection-relevant object",
        assessment_focus="perimeter activity, vehicles, groups of people, and unattended objects",
        recommendation="Maintain observation and add notes about direction, distance, and any perimeter concern.",
    ),
}


DEFAULT_PROFILE_NAME = "Search And Detect"


def get_profile(name: str) -> MissionProfile:
    return MISSION_PROFILES.get(name, MISSION_PROFILES[DEFAULT_PROFILE_NAME])
