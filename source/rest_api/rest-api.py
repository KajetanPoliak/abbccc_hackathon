from flask import Flask, jsonify, request
from result import ProjectDefinition, ProjectResult, SearchResults

app = Flask(__name__)

results = SearchResults()

results.Add(
    ProjectResult(
        ProjectDefinition(
            "Web Development",
            "Create a responsive website",
            "Frontend and backend implementation",
        ),
        "Project started",
        1634567890,
        "John Doe",
    )
)
results.Add(
    ProjectResult(
        ProjectDefinition(
            "Mobile App",
            "Develop a cross-platform mobile application",
            "UI/UX design and development",
        ),
        "In progress",
        1634657890,
        "Jane Smith",
    )
)
results.Add(
    ProjectResult(
        ProjectDefinition(
            "Data Analysis",
            "Analyze customer behavior data",
            "Data cleaning and visualization",
        ),
        "Completed",
        1634747890,
        "Bob Johnson",
    )
)


@app.route("/search", methods=["GET"])
def search() -> str:
    project_description = request.args.get("project_description", "").lower()
    project_definition = request.args.get("project_definition", "").lower()
    activity_description = request.args.get("activity_description", "").lower()
    name = request.args.get("name", "").lower()

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
        if name and name not in project.name.lower():
            continue

        response.Add(project)

    responseString: str = response.serialize()
    return responseString


if __name__ == "__main__":
    app.run(debug=True)
