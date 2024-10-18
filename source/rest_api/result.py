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
    ) -> None:
        self.project_description = project_definition
        self.project_definition = project_description
        self.activity_description = activity_description

    def to_dict(self) -> Dict[str, str]:
        return {
            "project_description": self.project_description,
            "project_definition": self.project_definition,
            "activity_description": self.activity_description,
        }

    def GetProjectDescription(self) -> str:
        return self.project_description

    def GetProjectDefinition(self) -> str:
        return self.project_definition

    def GetActivityDescription(self) -> str:
        return self.activity_description

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ProjectDefinition":
        return cls(
            project_description=data.get("project_description", ""),
            project_definition=data.get("project_definition", ""),
            activity_description=data.get("activity_description", ""),
        )


class ProjectResult:
    def __init__(
        self,
        project: ProjectDefinition,
        comment: str,
        datetime: datetime,
        name: str,
        duration: int,
    ) -> None:
        self.project = project
        self.comment = comment
        self.datetime = datetime
        self.name = name
        self.duration = duration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": self.project.to_dict(),
            "comment": self.comment,
            "datetime": self.datetime.isoformat(),
            "name": self.name,
            "duration": self.duration,
        }

    def GetProject(self) -> ProjectDefinition:
        return self.project

    def GetComment(self) -> str:
        return self.comment

    def GetDatetime(self) -> datetime:
        return self.datetime

    def GetName(self) -> str:
        return self.name

    def GetDuration(self) -> int:
        return self.duration

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectResult":
        return cls(
            project=ProjectDefinition.from_dict(data.get("project", {})),
            comment=data.get("comment", ""),
            datetime=datetime.fromisoformat(
                data.get("datetime", "1970-00-00T00:00:00")
            ),
            name=data.get("name", ""),
            duration=data.get("duration", 0),
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
