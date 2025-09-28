THE WHOLE FLOW OF THE APPLICATION


client
  │
  ▼
routers/story_router.py
  │   ├─> get_session_id() → cookie / uuid
  │   ├─> db: Session = Depends(get_db) → db/database.py
  │   ├─> StoryJob → models/job.py
  │   └─> StoryJobResponse → schemas/job.py
  │
  ▼
background_tasks.generate_story_task
  │   ├─> SessionLocal() → db/database.py
  │   ├─> StoryGenerator.generate_story() → core/story_generator.py
  │   │      └─> prompts/templates
  │   └─> writes Story + StoryNodes → models/story.py
  │
  └─> update StoryJob status → models/job.py
  │
  ▼
client GET /stories/{story_id}/complete
  │
  ├─> router → get_complete_story()
  │       └─> fetch Story + StoryNodes → models/story.py
  │
  └─> build_complete_story_tree() → schemas/story.py → JSON response
