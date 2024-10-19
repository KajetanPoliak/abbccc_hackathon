import json
from datetime import datetime

import plotly.graph_objects as go
import requests
import streamlit as st
from result import ProjectDefinition, ProjectResult, SearchResults


def search_projects(
    project_description: str,
    project_definition: str,
    activity_description: str,
    subject: str,
) -> SearchResults:
    base_url = "http://localhost:5000"  # Assuming Flask runs on default port
    params = {
        "project_description": project_description,
        "project_definition": project_definition,
        "activity_description": activity_description,
        "subject": subject,
    }
    response = requests.get(f"{base_url}/search", params=params)
    return SearchResults.deserialize(response.text)


def split_by_project(results: SearchResults) -> dict[str, list[ProjectResult]]:
    resultDict: dict[str, list[ProjectResult]] = {}
    for project in results.Items():
        definition: str = project.GetProject().GetProjectDescription()
        if definition in resultDict:
            resultDict[definition].append(project)
        else:
            resultDict[definition] = [project]
    return resultDict


def split_by_description(
    projects: list[ProjectResult],
) -> dict[str, list[ProjectResult]]:
    resultDict: dict[str, list[ProjectResult]] = {}
    for project in projects:
        description: str = project.GetProject().GetProjectDefinition()
        if description in resultDict:
            resultDict[description].append(project)
        else:
            resultDict[description] = [project]
    return resultDict


def split_by_activity(
    projects: list[ProjectResult],
) -> dict[str, list[ProjectResult]]:
    resultDict: dict[str, list[ProjectResult]] = {}
    for project in projects:
        activity: str = project.GetProject().GetActivityDescription()
        if activity in resultDict:
            resultDict[activity].append(project)
        else:
            resultDict[activity] = [project]
    return resultDict


def get_total_duration(projects: list[ProjectResult]) -> int:
    result: int = 0
    for project in projects:
        result += project.GetDuration()
    return result


def userIdToName(id: str) -> str:
    # in current calendar we do not have access to the attendees :(
    return "Aleksandar CEBZAN"


def create_project_histogram(project_results: list[ProjectResult]) -> go.Figure:
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    hours_by_day = {}

    for day in days:
        hours_by_day[day] = 0

    for result in project_results:
        day = result.GetDatetime().strftime("%A")
        hours = result.GetDuration()
        hours_by_day[day] += hours

    x = days
    y = [hours_by_day[day] for day in days]

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(
        title="Hours Spent by Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Hours Spent",
        height=400,
    )
    return fig


st.title("Project Search")

project_description = st.text_input("Project Description")
project_definition = st.text_input("Project Definition")
activity_description = st.text_input("Activity Description")
name = st.text_input("Subject")

if st.button("Search"):
    results = search_projects(
        project_description, project_definition, activity_description, name
    )

    if results.Items():
        project_groups = split_by_project(results)
        for project, project_results in project_groups.items():
            project_duration = get_total_duration(project_results)
            st.markdown(f"## Project: {project}")
            st.markdown(f"**Total Duration**: {project_duration}")
            fig = create_project_histogram(project_results)
            st.plotly_chart(fig)

            description_groups = split_by_description(project_results)
            for description, description_results in description_groups.items():
                description_duration = get_total_duration(description_results)
                st.markdown(f"### Description: {description}")
                st.markdown(f"**Duration**: {description_duration}")

                activity_groups = split_by_activity(description_results)
                for activity, activity_results in activity_groups.items():
                    activity_duration = get_total_duration(activity_results)
                    with st.expander(
                        f"Activity: {activity} (Duration: {activity_duration})"
                    ):
                        for result in activity_results:
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
                                    <p><strong>Datetime:</strong> {result.GetDatetime()}</p>
                                    <p><strong>User:</strong> {userIdToName(result.GetUserId())}</p>
                                    <p><strong>Subject:</strong> {result.GetSubject()}</p>
                                    <p><strong>Body:</strong> {result.GetBody()}</p>
                                    <p><strong>Duration:</strong> {result.GetDuration()}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
            st.markdown("---")  # Add a horizontal line between projects
    else:
        st.write("No results found.")
