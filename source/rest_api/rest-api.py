import json
import os
from datetime import date, datetime

from flask import Flask, jsonify, request
from result import ProjectDefinition, ProjectResult, SearchResults

app = Flask(__name__)

with open("./data/data_results.json", encoding="utf-8") as f:
    data = json.load(f)

results = SearchResults()

for event in data:
    for dt_instance in event["instances"]:
        date_start = date.fromisoformat(dt_instance)
        time_start = datetime.fromisoformat(event["start"])
        results.Add(
            ProjectResult(
                project=ProjectDefinition(
                    project_description=event["project_description"],
                    project_definition="yy",  # event["project_definition"]
                    activity_description=event["project_activity"],
                    confidence=0.99,
                ),
                datetime_start=datetime(
                    year=date_start.year,
                    month=date_start.month,
                    day=date_start.day,
                    hour=time_start.hour,
                    minute=time_start.hour,
                    second=time_start.second,
                ),
                user_id=event["id"],
                duration=event["duration"],
                subject=event["subject"],
                body=event["body_preview_clean"],
            )
        )


@app.route("/search", methods=["GET"])
def search() -> str:
    project_description = request.args.get("project_description", "").lower()
    project_definition = request.args.get("project_definition", "").lower()
    activity_description = request.args.get("activity_description", "").lower()
    user_id = request.args.get("user_id", "").lower()

    response: SearchResults = SearchResults()

    for project in results.Items():
        if (
            project_description
            and project_description
            not in project.project.project_description.lower()
        ):
            continue
        if (
            project_definition
            and project_definition
            not in project.project.project_definition.lower()
        ):
            continue
        if (
            activity_description
            and activity_description
            not in project.project.activity_description.lower()
        ):
            continue
        if user_id and user_id not in project.name.lower():
            continue

        response.Add(project)

    responseString: str = response.serialize()
    return responseString


if __name__ == "__main__":
    app.run(debug=True)
