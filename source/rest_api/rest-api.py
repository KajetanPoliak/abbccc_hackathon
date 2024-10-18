from flask import Flask, jsonify, request
from result import ProjectDefinition, ProjectResult, SearchResults

app = Flask(__name__)

results = SearchResults()


@app.route("/search", methods=["GET"])
def search() -> str:
    project_description = request.args.get("project_description", "").lower()
    project_definition = request.args.get("project_definition", "").lower()
    activity_description = request.args.get("activity_description", "").lower()
    name = request.args.get("name", "").lower()

    response: SearchResults = SearchResults()

    for project in results:
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
        if name and name not in project.name.lower():
            continue

        response.Add(project)

    return ""


if __name__ == "__main__":
    app.run(debug=True)
