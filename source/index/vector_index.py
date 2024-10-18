import json
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import numpy.typing as npt
import pandas as pd
from transformers import AutoModel, AutoTokenizer

from source.utils.logging import get_stream_logger
from source.vectorization.deep import encode_documents, get_tokenizer_and_model

__here__ = Path(__file__).resolve().parent
__root__ = __here__.parents[1]
__data_dir__ = __root__ / "data"
assert __data_dir__.exists(), f"Data directory not found: {__data_dir__!r}"


class FaissIndex:
    def __init__(self, dim: int):
        self.logger = get_stream_logger(self.__class__.__name__)
        self.index = faiss.IndexFlatIP(dim)
        self.index_items: List[str] = []

    def set_embeddings(self, embeddings: npt.NDArray, items: List[str]) -> None:
        self.index.reset()
        self.logger.info(f"Setting embeddings with shape: {embeddings.shape}")
        self.index.add(embeddings)
        self.index_items = items

    def search_by_vector_query(
        self,
        query: npt.NDArray,
        k: Optional[int] = 5,
    ) -> Tuple[npt.NDArray, npt.NDArray, List[str]]:
        if k is None:
            k = len(self.index_items)
        distances, indices = self.index.search(query, k)
        return distances, indices, [self.index_items[i] for i in indices[0]]

    @classmethod
    def process_query_document(cls, title: str, body: str) -> str:
        """Process a query document and extract its embeddings from the title
        and body"""
        query = f"{title} / {body}".strip("/ ")
        return query

    def to_dataframe(
        self, score: npt.NDArray, idx: npt.NDArray, items: List[str]
    ) -> pd.DataFrame:
        data = [score, idx, items]
        kk = [data[0][0], data[1][0], data[2]]
        df = pd.DataFrame(kk, index=["context_score", "indexes", "Project"]).T
        df[["project", "activity"]] = df["Project"].str.split(": ", expand=True)
        df.drop(columns=["Project", "indexes"], inplace=True)
        return df

    def search(
        self,
        document: str,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        k: Optional[int] = None,
        device: str = "cpu",
    ) -> Tuple[npt.NDArray, npt.NDArray, List[str]]:
        query = encode_documents(
            [document], tokenizer, model, normalize=True, device=device
        )
        # Normalize the query vector
        query /= np.linalg.norm(query, axis=1).reshape(-1, 1)
        return self.search_by_vector_query(query, k)

    def save(self, filename: str) -> None:
        path = __data_dir__ / filename
        self.logger.info(f"Saving index to {path!r}")
        faiss.write_index(self.index, str(path))
        # Save the index items
        path_json = __data_dir__ / f"{filename}.json"
        self.logger.info(f"Saving index items to {path!r}")
        with open(path_json, "w") as f:
            data = json.dumps(self.index_items, indent=4)
            f.write(data)

    @classmethod
    def from_file(cls, filename: str) -> "FaissIndex":
        instance = cls(dim=0)
        instance.index = faiss.read_index(str(__data_dir__ / filename))
        with open(__data_dir__ / f"{filename}.json", "r") as f:
            instance.index_items = json.load(f)
        return instance


if __name__ == "__main__":
    index = FaissIndex(dim=768)
    _sample_vectors = np.random.randn(100, 768).astype("float32")
    _sample_vectors /= np.linalg.norm(_sample_vectors, axis=1).reshape(-1, 1)
    index.set_embeddings(
        embeddings=_sample_vectors, items=[f"item_{i}" for i in range(100)]
    )
    # Search for the most similar document for vector at index 0
    dist, idx, itms = index.search_by_vector_query(_sample_vectors[:1], k=None)
    print("First search:", dist, idx, itms)
    assert idx[0][0] == 0 and np.isclose(dist[0][0], 1.0, atol=1e-3)

    # Save the index
    index.save("faiss_ip.index")

    # Load the index
    index = FaissIndex.from_file("faiss_ip.index")
    dist, idx, itms = index.search_by_vector_query(_sample_vectors[:1], k=5)
    print("Search after the load:", dist, idx, itms)
    assert idx[0][0] == 0 and np.isclose(dist[0][0], 1.0, atol=1e-3)

    # Read project data to create an index
    import pandas as pd

    df = pd.read_csv(__data_dir__ / "trimmed_project_data.csv").fillna("")
    df = df.groupby(
        by=["Project Description", "Activity Description"],
        as_index=False,
    ).agg({"Comment": lambda item: " | ".join(item.tolist()).strip("| ")})
    tok, mod = get_tokenizer_and_model(device="mps")
    # Create a list of documents by concatenating project name,
    # activity description, and comment
    documents = df.apply(
        lambda row: f"{row['Project Description']}: "
        f"{row['Activity Description']} / "
        f"{row['Comment']}".strip("/ "),
        axis=1,
    ).tolist()
    document_descriptions = [
        """Developing and maintaining tactical scenario guides by thoroughly
        reading and reviewing comprehensive
documentation to ensure accuracy, consistency, and compliance with
established protocols, incorporating feedback
from subject matter experts and adhering to organizational standards for
military training simulations.""",
        """Conducting a structured internal kickoff meeting to initiate
        communication among key stakeholders, including
Joint Training (JT), Justin Beaver (JB), and Richard, to discuss project
objectives, timelines, and technical
requirements for the integration of military training simulations,
facilitated by an Internal Business Analyst.""",
        """Implementing a comprehensive software deployment project,
        involving the installation of OptiMax v1.22 in Azure,
        integration of temporary licensing, and configuration of user groups
        and accounts for the Norwegian market,
        including a kickoff meeting with key stakeholders, setup of DK Cloud
        environment, and collaboration to ensure
        seamless adoption by Norfors users, facilitated by technical
        expertise.""",
        """Designing, developing, and testing a cutting-edge software
        solution, Ene Vision, incorporating innovative
features and functionalities to enhance operational efficiency,
user experience, and data analysis capabilities,
with a focus on research and development, quality assurance, and deployment
readiness.""",
        """Creating a comprehensive set of documentation for the Ene Vision
        software solution, including Functional Design
        Specifications (FDS), to provide clear and detailed information on
        system architecture, components, interfaces,
        and user interactions, ensuring consistency and accuracy across all
        development phases.""",
        """Designing, implementing, and configuring dashboards for the Ene
        Vision software solution to provide real-time
        insights and monitoring capabilities, and addressing a critical
        requirement by manually adding missing input data
        to enable seamless functionality, ensuring data completeness and
        accuracy, and preparing the system for further
        development and integration.""",
        """Convening a project meeting with key stakeholders, including Jan
        Bitta, to discuss and address critical issues,
        implement required changes, and align on next steps for the Ene
        Vision software solution, facilitating effective
        communication, resolving outstanding problems, and ensuring timely
        progress towards project goals.""",
        """Identifying and resolving a technical issue related to the
        simulation process of the Ene Vision software
        solution, specifically addressing the presence of Not-a-Number (NaN)
        values in input data, which is hindering
        accurate results, and implementing necessary fixes, such as data
        cleaning, validation, or recalibration, to
        restore reliable performance and ensure high-quality simulations.""",
        """Conducting a comprehensive Testing & Handover phase for the Ene
        Vision software solution, including a Full
        Acceptance Test (FAT) day, addressing technical issues such as VPN
        connectivity and data exchange errors with
        JEPX, correcting problems with input and output files, implementing
        fixes for script-related issues, and debugging
        correction plans to ensure seamless handover to end-users,
        facilitated by thorough testing, quality assurance, and
        collaborative communication with key stakeholders.""",
        """Attending a Key Opinion (KO) meeting for the Energy Efficiency
        module of the GTB EMS Optimax system, gathering
        input from technical experts to refine and validate system
        requirements, confirm implementation timelines, and
        ensure alignment with industry standards and best practices, enabling
        informed decision-making and successful
        project execution.""",
        """Installing a virtual machine (VM) environment for the Finland Optimax Support 2024 system in Microsoft Azure,
ensuring seamless scalability, high availability, and secure deployment of critical software components, including
configuration of cloud infrastructure, VM setup, and initial testing to validate functionality and prepare for
future maintenance and support.""",
        """Conducting a comprehensive set of engineering tasks for the Gasum Loiste system, including:

* Parsing and importing SAF files into the database
* Developing data simulation scripts and dashboards
* Creating FDS documentation and preparing data and case scenarios
* Installing FireEye on virtual machines (devOps) and documenting its configuration
* Preparing Gasum Dashboards and checking progress
* Implementing IAT/FAT/SAT documentation and meetings
* Testing SAF Heat Storage value conversion and FAT preparation
* Debugging optimizer scripts and creating new ones for 6 different files from SFTP
* Importing Loiste data and scripting for SAF export
* Preparing FAT, exporting data to sandbox, and updating Gasum project

Additionally, tasks such as:

* Backing up old Sandbox and preparing new VMs
* Creating SAT documentation and preparation documents
* Testing Gasum sandbox update to 6.4
* Holding meetings for the new Gasum project and updating documentation
* Completing production VM setup
* Finalizing SAT documentation and completion of project documentation

 Ensuring a thorough and meticulous approach to deliver the Gasum Loiste system, with a focus on quality, testing,
and collaboration.""",
        """Installing and configuring Advanced Authentication and Authorization (AAD) Single Sign-On (SSO) on Gasum's OPX
system in their Azure environment, resolving a technical issue to enable seamless authentication, followed by the
installation of Modelica and Optimax Documentation, and then creating a virtual machine (VM) for Optimax
deployment.

Additionally:

* Configuring AAD SSO on a test VM to facilitate troubleshooting and testing
* Setting up SFTP configuration for secure data transfer
* Resolving issues with AAD SSO in Gasum Azure, including configuring it for use with UPN (User Principal Name)

This includes tasks such as:

* Troubleshooting and resolving AAD SSO problems on the OPX system
* Installing and configuring Modelica and Optimax Documentation
* Creating a test VM for Optimax deployment and installation of AAD SSO
* Configuring SFTP settings for secure data transfer

Ensuring a smooth and efficient setup of AAD SSO, Modelica, and Optimax Documentation on Gasum's Azure
environment, with a focus on security, authentication, and data transfer.""",
        """Holding a series of meetings to facilitate collaboration, project planning, and progress updates for the Gasum
Loiste project, including:

* Initial meetings with Gasum and FIABB (Finnish Association of Oil and Gas Industries) to discuss project scope,
objectives, and timelines
* Multiple meetings with Gasum and FIABB to review project progress, address concerns, and provide updates on
technical developments
* Meetings with Richard about the Model to ensure alignment and understanding of its application
* Meetings with Joonas to cover specific topics and provide clarification on project requirements
* Follow-up meeting with FIABB to confirm next steps and resolve any outstanding issues
* Meeting with the EC (Engineering Competence) team from DevOps to discuss technical aspects, infrastructure, and
deployment plans

These meetings aimed to:

* Establish clear communication channels between stakeholders
* Ensure alignment of project objectives and timelines
* Provide regular updates on project progress and milestones
* Address concerns and resolve issues in a timely manner
* Foster collaboration and knowledge sharing among team members

By holding these meetings, the project teams could work together efficiently, address challenges promptly, and
ensure the successful delivery of the Gasum Loiste system.""",
        """Implementing the necessary engineering tasks for the NUS Umea Sweden project, including:

* Installing Modelica and Node-RED with Python on a Tomtebostrand virtual machine (VM)
* Modifying the Node-RED package to utilize Python, ensuring seamless integration with other components
* Resolving issues with Access to AAD SSO for the Tomtebo VM, enabling secure authentication
* Troubleshooting resolver issues on the Tomtebo VM, ensuring accurate and reliable data processing
* Creating a new VHD (Virtual Hard Disk) for SEABB Azure, enabling deployment of the OPX 6.4 software for NUS
Prestudy
* Installing and configuring OPX 6.4 for NUS Prestudy, ensuring successful operation and integration with existing
systems

These tasks aimed to:

* Develop and integrate Modelica and Node-RED with Python for efficient data processing and automation
* Enhance security and access control by resolving AAD SSO issues
* Troubleshoot and resolve resolver issues on the Tomtebo VM, ensuring accurate data processing
* Set up a new VHD for SEABB Azure, enabling deployment of OPX 6.4 software
* Ensure successful installation and configuration of OPX 6.4 for NUS Prestudy

By completing these engineering tasks, the project team was able to deliver a functional and integrated solution
that met the requirements of the NUS Umea Sweden project.""",
        """Delivering a series of tasks to support the OPTIMAX APPS&MODELS CZ 2024 WP-12081.04 project, including:

* Setting up demo virtual machines (VMs) for DE ABB in Azure, providing a testing environment for Optimax
* Assisting Bernhard and Anton with installation and configuration of OPX 6.4.0 on multiple platforms
* Completing the H2 P2X module without ABB requirements, ensuring successful deployment
* Resolving issues with H2 Valleys (DE ABB) to ensure accurate functionality
* Configuring DEMO VMs for ITABB, including setting up Optimax demo environment
* Verifying every VM in the DE ABB Azure subscription to ensure OS version and OPX version are compatible
* Adding a new VHD file for OPX 6.4 to Azure, enabling deployment of the latest software
* Setting up AAD SSO (Active Directory Single Sign-On) for external users on the HPP Sizing tool, ensuring secure
access
* Troubleshooting issues with HPP Sizing tool, resolving problems and implementing fixes
* Preparing a new OPX ISO file for uploading to Azure on July 19, 2024, ensuring timely deployment
* Creating new VMs for ITABB demo, including adding a new P2X module
* Resolving malware issues on EUOPC VM, ensuring the environment is secure and safe
* Upgrading OPX-hynamics-upgrade VM to ensure compatibility with latest software versions
* Verifying the status of upgraded VM to ensure successful deployment

These tasks aimed to:

* Provide a stable and functional demo environment for DE ABB in Azure
* Ensure successful installation and configuration of OPX 6.4 on multiple platforms
* Complete critical modules, such as H2 P2X, without requiring ABB support
* Troubleshoot and resolve issues with H2 Valleys and HPP Sizing tool
* Set up secure access for external users using AAD SSO
* Ensure timely deployment of new software versions, including OPX 6.4
* Deliver a high-quality, malware-free environment for EUOPC VM

By completing these tasks, the project team was able to deliver a comprehensive and functional solution that met
the requirements of the OPTIMAX APPS&MODELS CZ 2024 WP-12081.04 project.""",
        """Implementing the Optimax system for ECOPETROL REFICAR Refinery in Colombia, including:

* Installing the Optimax system to support refinery operations and process optimization
* Configuring the system to meet specific refinery requirements and regulations
* Setting up a Virtual Private Network (VPN) to enable secure remote access to the Optimax system

These tasks aimed to:

* Ensure the Optimax system is properly installed, configured, and integrated with existing refinery systems
* Provide a secure and reliable means of accessing the Optimax system from remote locations, enabling real-time
monitoring and control of refinery operations
* Meet regulatory requirements for security and data protection in the refinery industry

The installation process likely involved:

* Coordinating with ECOPETROL REFICAR personnel to understand specific system requirements and integration needs
* Installing the Optimax hardware and software components, including servers, databases, and network
infrastructure
* Configuring the VPN to connect remote users to the Optimax system securely
* Testing and validating the installation to ensure seamless operation and access to the system

The resulting installation will provide ECOPETROL REFICAR with a robust and secure Optimax system for process
optimization, real-time monitoring, and control of refinery operations. The VPN setup ensures that authorized
personnel can access the system from remote locations, while maintaining strict security measures.""",
        """Conducting testing and handover activities for the Optimax system at ECOPETROL REFICAR Refinery in Colombia,
including:

* Testing the Optimax system to ensure it meets specific refinery requirements and is functioning as expected
* Validating system performance, functionality, and integration with existing systems
* Identifying and addressing any issues or discrepancies during testing

Additional activities included:

* Attempting to install SiteEMS (Station Management System) on site, which may involve:
	+ Coordinating with ECOPETROL REFICAR personnel to understand their requirements and expectations for the system
	+ Installing and configuring SiteEMS software and hardware components
	+ Testing and validating the installation to ensure seamless operation and integration with Optimax

The handover process involved:

* Providing training and support to ECOPETROL REFICAR personnel on the Optimax system and its features
* Ensuring that personnel are familiar with system functionality, configuration, and maintenance procedures
* Coordinating with ECOPETROL REFICAR personnel to schedule regular system checks and maintenance activities

The testing and handover process aimed to:

* Ensure the Optimax system meets ECOPETROL REFICAR's requirements and is functioning as expected
* Provide a smooth transition of ownership and operation of the system
* Ensure that ECOPETROL REFICAR personnel have the necessary knowledge and skills to operate and maintain the
Optimax system

By completing these activities, the team can ensure a successful implementation of the Optimax system at ECOPETROL
REFICAR Refinery in Colombia.""",
        """Implementation of Optimax support for P2X Solutions, involving:

* Creation and upload of a Virtual Hard Disk (VHD) file to facilitate the deployment of Optimax software
* Successful deployment of Optimax on the specified virtual environment, ensuring seamless integration with
existing systems and configurations

Key aspects of this project include:
- Technical expertise in creating and uploading VHD files for efficient software deployment.
- Strategic planning and execution of the Optimax deployment process to ensure minimal disruption to existing
operations.

This project involves optimizing the workflow for P2X Solutions' clients by streamlining the software upload and
installation process.""",
        """Implementation of OPTIMAX solution for Senatobia Extension, involving:

* Initial technical preparation and planning phase, including:
	+ Reviewing available documentation to ensure thorough understanding of the system requirements
	+ Conducting an internal kick-off meeting to align project stakeholders and set clear objectives
	+ Preparing presentation materials and content for key stakeholders, including a comprehensive overview of the
OPTIMAX solution

Subsequent phases include:
* Finalization of detailed documentation and preparation for key stakeholders, including a thorough understanding
of system configurations and settings
* Conducting pre-internal KO (Kick-off) meeting to confirm project timelines and milestones
* Preparing presentation materials and content for key stakeholders, including Senatobia staff and USABB
representatives

The project also involves:
* Regular meetings with Senatobia team members to discuss progress, address concerns, and provide updates on the
OPTIMAX solution
* Establishing communication channels with the customer through bi-weekly meetings to ensure alignment with
project objectives and timelines
* Integration of MODBUS connectivity with the weather station to enable seamless data exchange and real-time
monitoring

This project requires careful planning, coordination, and execution to ensure successful deployment and
utilization of the OPTIMAX solution for Senatobia Extension.""",
        """Implementation of Optimax system for Shell ETCA EMS, involving:

* Technical collaboration with Shell to discuss and finalize details of the Optimax system, including:
	+ Meeting with Shell's Load Scheduler team to align on technical requirements and ensure seamless integration with
existing systems
	+ Shell meeting to review project objectives, timelines, and expectations, and to address any questions or
concerns

These meetings aim to:
* Ensure a deep understanding of Shell's operational needs and requirements for the Optimax system
* Establish clear communication channels between the Shell team and the implementation team to facilitate a smooth
project execution
* Validate the technical feasibility of the Optimax solution and identify potential areas for optimization or
improvement

By working closely with Shell to address their specific needs and requirements, we can ensure that the Optimax
system is tailored to meet their operational demands and deliver value to their business.""",
        """Installation of Optimax system for Shell ETCA EMS, involving:

* Virtualization infrastructure setup:
	+ Creation of a virtual machine (VM) specifically designed for Optimax software deployment
	+ Installation of OPTIMAX on the newly created VM, ensuring seamless integration with existing systems
* Integration with other technologies:
	+ Node-RED Python installation to facilitate data exchange and automation between systems
	+ OPX 6.4 VM installation on a separate virtual environment for Shell's use, ensuring compatibility and security

Critical tasks include:
* Backup of the old VM to ensure data integrity and minimize downtime during the transition
* Deployment of the new VM, including configuration and setup of network connectivity
* Configuration of the VM's network settings to ensure optimal communication between systems

These installation steps aim to:
* Ensure a stable and secure environment for Optimax software deployment
* Facilitate integration with existing systems and technologies
* Provide a scalable and flexible infrastructure for Shell ETCA EMS operations""",
    ]

    embs = encode_documents(
        document_descriptions,
        tokenizer=tok,
        model=mod,
        normalize=True,
        device="mps",
    )
    index = FaissIndex(dim=embs.shape[1])
    # Create the index items by preserving only the project name and activity
    # descriptions
    items_parsed = [document.split("/")[0].strip() for document in documents]
    index.set_embeddings(embs, items=items_parsed)
    index.save("faiss_ip.index")

    # Example search
    title = "Discussing Optimax"
    body = "This is a discussion about Optimax"
    query = FaissIndex.process_query_document(title, body)
    dist, idx, itms = index.search(query, tok, mod, k=None, device="mps")
    print("Example search:", dist, idx, itms)
