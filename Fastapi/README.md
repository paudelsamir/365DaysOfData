# Day 1: Introduction to FastAPI

Started exploring FastAPI as a modern, fast web framework for building APIs with Python. Learned about its basic concepts and why it's becoming popular for backend development.
Focused on the specific advantages of FastAPI for machine learning deployments, including its performance characteristics and automatic documentation features.
![API](images/api.png)
![Why Fast to Run](images/02_why_fast_to_run.png)


# Day 2: FastAPI Documentation Features

Explored FastAPI's automatic documentation generation with Swagger UI and ReDoc, which makes testing and sharing APIs much simpler.
![FastAPI Doc](images/03_fastapi_doc.png)

# Day 3: Building a Simple App

Worked through a blog guide on building a basic application with FastAPI, implementing fundamental concepts like routing and response handling.
![Blog Build an App with FastAPI](images/03_blog_build__an_app_with_fastapi.png)

# Day 4: HTTP Request Methods

Studied HTTP request methods (GET, POST, PUT, DELETE) and how they're implemented in FastAPI for different API operations.
![HTTP Request Methods](images/04_http_request_methods.png)

# Day 5: REST Architecture Principles Async/Await in FastAPI

Learned about RESTful API design principles and how FastAPI encourages following these conventions for building maintainable APIs.
Investigated how FastAPI leverages Python's async capabilities for handling concurrency and improving performance in high-load scenarios.
![Async Await](images/04_async_await.png)
![REST Architecture](images/04_rest_architecture.png)


# Day 6: Path and Query params, Request Bodies in FastAPI

Practiced implementing and working with different parameter types in FastAPI endpoints, focusing on path and query parameters.
Learned how to handle request bodies in FastAPI using Pydantic models for validation and automatic conversion of incoming JSON data.
![Request Body](images/05_request_body.png)
![Path Parameter](images/05_path_parameter.png)
![Query Parameter](images/05_query_parameter.png)


# Day 7: Mini Project with FastAPI - Implementing GET and PUT Operations

Built a comprehensive REST API with FastAPI as a mini-project, implementing GET and PUT endpoints with proper data validation. This hands-on exercise reinforced my understanding of request handling, response models, and API design principles essential for ML model deployment.
![API Implementation](images/06_apis.png)

# Day 8: Completing CRUD Operations with UPDATE and DELETE

Extended the mini-project by implementing UPDATE (PATCH) and DELETE operations, completing the full CRUD functionality. Focused on proper status code handling, error responses, and ensuring data consistency across operations. This exercise provided valuable experience with FastAPI's dependency injection system.
![CRUD Implementation Code](images/06_code.png)

# Day 9: Building Industry-Ready APIs with FastAPI

Explored best practices for creating production-grade APIs with FastAPI, including proper error handling, authentication, rate limiting, and documentation. Learned how to structure larger applications using APIRouter and dependency injection patterns that align with industry standards.
![Industry-Level ML with FastAPI](images/07_ml_with_fastapi_industry_level_app.png)

# Day 10: Containerizing FastAPI Applications

Learned how to package FastAPI applications using Docker containers, ensuring consistent deployment across different environments. Explored multi-stage builds for optimized images, environment configuration, and container orchestration concepts for scalable API deployment.
![FastAPI in Containers](images/08_fastapi_in_containers.png)

# Day 11: Project Setup - Choose Your Own Adventure

Started a new project implementing a "Choose Your Own Adventure" application with FastAPI as the backend. Set up the development environment and explored UV as a faster alternative to traditional Python package managers for dependency installation.
![Project Setup Backend FastAPI](images/10_project_setup_backend_fastapi.png)
![UV Package Manager](images/10_uv.png)

# Day 12: Database Design and Core Components

Designed core configuration handlers for application settings and structured the database using SQLAlchemy. Created database models to represent tables and relationships, and developed Pydantic schemas to define API input/output formats for the adventure story application.
![Core Configs Models Code](images/11_core_configs_models_code.png)
![Database Python Code](images/11_databasepython_code.png)
![Models Jobs Story Code](images/11_models_jobs_story_code.png)
![Schemas Job Story](images/11_schemas_job_story.png)

# Day 13: API Implementation and Background Tasks

Implemented routers for jobs and stories endpoints to handle the core functionality of the adventure application. Integrated background tasks for asynchronous story generation, enabling non-blocking API responses while stories are being created.
![Router Job Endpoint](images/12_router_job_endpoint.png)
![Router Story Endpoints Post Get](images/12_router_story_endpoints_post_get.png)

# Day 14: Backend Completion and Debugging

Resolved numerous debugging challenges to get all APIs working properly. The most challenging aspect was connecting Ollama (local LLM) and parsing its output into structured data for the application. Successfully integrated the LLM component with the backend services.
![Completed Backend](images/13_completed_backend.png)
![Database Troubleshoot Preview](images/13_database_troubleshoot_preview.png)
![Sample Output Underwater World Topic](images/13_sample_outpu_of_underwater_world_topic.png)

# Day 15: Frontend Integration and Project Completion

Designed and implemented the frontend interface for the Choose Your Own Adventure application. Polished the user flow to ensure a seamless experience. While the application worked well locally, deployment was limited by the decision to use Ollama rather than cloud-based solutions like OpenAI. Successfully completed the project with a fully functional local demonstration.
![Project Demo](images/14_project_demo.png)


Complete REPO of the project: [github.com/paudelsamir/Choose-Your-Own-Adventure](https://github.com/paudelsamir/Choose-Your-Own-Adventure)