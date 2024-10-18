import json
from datetime import datetime
from typing import Any, Dict, List


# definition of the result structure we will save for rest-api
class ProjectDefinition:
    def __init__(
        self,
        project_description: str,
        project_definition: str,
        activity_description: str,
        confidence: float,
    ) -> None:
        self.project_description = project_definition
        self.project_definition = project_description
        self.activity_description = activity_description
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_description": self.project_description,
            "project_definition": self.project_definition,
            "activity_description": self.activity_description,
            "confidence": self.confidence,
        }

    def GetProjectDescription(self) -> str:
        return self.project_description

    def GetProjectDefinition(self) -> str:
        return self.project_definition

    def GetActivityDescription(self) -> str:
        return self.activity_description

    def GetConfidence(self) -> float:
        return self.confidence

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectDefinition":
        return cls(
            project_description=data.get("project_description", ""),
            project_definition=data.get("project_definition", ""),
            activity_description=data.get("activity_description", ""),
            confidence=data.get("confidence", -1),
        )


class ProjectResult:
    def __init__(
        self,
        project: ProjectDefinition,
        datetime_start: datetime,
        user_id: str,
        duration: int,
        subject: str,
        body: str,
    ) -> None:
        self.project = project
        self.datetime_start = datetime_start
        self.user_id = user_id
        self.duration = duration
        self.subject = subject
        self.body = body

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": self.project.to_dict(),
            "datetime_start": self.datetime_start.isoformat(),
            "user_id": self.user_id,
            "duration": self.duration,
            "subject": self.subject,
            "body": self.body,
        }

    def GetProject(self) -> ProjectDefinition:
        return self.project

    def GetDatetime(self) -> datetime:
        return self.datetime_start

    def GetUserId(self) -> str:
        return self.user_id

    def GetDuration(self) -> int:
        return self.duration

    def GetSubject(self) -> str:
        return self.subject

    def GetBody(self) -> str:
        return self.body

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectResult":
        return cls(
            project=ProjectDefinition.from_dict(data.get("project", {})),
            datetime_start=datetime.fromisoformat(
                data.get("datetime_start", "1970-00-00T00:00:00")
            ),
            user_id=data.get("user_id", ""),
            duration=data.get("duration", 0),
            subject=data.get("subject", ""),
            body=data.get("body", ""),
        )


class SearchResults:
    def __init__(self) -> None:
        self.results: list[ProjectResult] = []

    def Add(self, result: ProjectResult) -> None:
        self.results.append(result)

    def serialize(self) -> str:
        return json.dumps(
            {"results": [result.to_dict() for result in self.results]}, indent=2
        )

    def Items(self) -> list[ProjectResult]:
        return self.results

    @classmethod
    def deserialize(cls, json_str: str) -> "SearchResults":
        data = json.loads(json_str)
        search_results = cls()
        for result_data in data.get("results", []):
            search_results.Add(ProjectResult.from_dict(result_data))
        return search_results
