import json
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

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ProjectDefinition":
        return cls(
            project_description=data.get("project_description", ""),
            project_definition=data.get("project_definition", ""),
            activity_description=data.get("activity_description", ""),
        )


class ProjectResult:
    def __init__(
        self, project: ProjectDefinition, comment: str, time: int, name: str
    ) -> None:
        self.project = project
        self.comment = comment
        self.time = time
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": self.project.to_dict(),
            "comment": self.comment,
            "time": self.time,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectResult":
        return cls(
            project=ProjectDefinition.from_dict(data.get("project", {})),
            comment=data.get("comment", ""),
            time=data.get("time", 0),
            name=data.get("name", ""),
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
