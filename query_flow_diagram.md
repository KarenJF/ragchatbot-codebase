# RAG Chatbot Query Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend<br/>(script.js)
    participant API as FastAPI<br/>(app.py)
    participant RAG as RAGSystem<br/>(rag_system.py)
    participant Session as SessionManager
    participant AI as AIGenerator<br/>(ai_generator.py)
    participant Claude as Anthropic Claude
    participant Tools as ToolManager<br/>(search_tools.py)
    participant Vector as VectorStore<br/>(ChromaDB)

    User->>Frontend: Types query & clicks send
    Frontend->>Frontend: Disable input, show loading
    
    Frontend->>+API: POST /api/query<br/>{query, session_id}
    
    API->>API: Create session if null
    API->>+RAG: query(query, session_id)
    
    RAG->>Session: get_conversation_history(session_id)
    Session-->>RAG: previous messages
    
    RAG->>+AI: generate_response()<br/>(query, history, tools, tool_manager)
    
    AI->>AI: Build system prompt + context
    AI->>+Claude: messages.create()<br/>(with tools enabled)
    
    alt Claude decides to use tools
        Claude-->>AI: tool_use response
        AI->>+Tools: execute_tool(search_course_content, params)
        
        Tools->>+Vector: search(query, course_name, lesson_number)
        Vector->>Vector: Semantic search in ChromaDB
        Vector-->>-Tools: SearchResults (docs + metadata)
        
        Tools->>Tools: Format results with course/lesson context
        Tools->>Tools: Store sources in last_sources
        Tools-->>-AI: Formatted search results
        
        AI->>+Claude: messages.create()<br/>(with tool results)
        Claude-->>-AI: Final response text
    else Claude answers directly
        Claude-->>-AI: Direct response text
    end
    
    AI-->>-RAG: Generated response
    
    RAG->>Tools: get_last_sources()
    Tools-->>RAG: [source list]
    RAG->>Tools: reset_sources()
    
    RAG->>Session: add_exchange(session_id, query, response)
    
    RAG-->>-API: (response, sources)
    
    API->>API: Create QueryResponse model
    API-->>-Frontend: JSON: {answer, sources, session_id}
    
    Frontend->>Frontend: Remove loading message
    Frontend->>Frontend: Render markdown response + sources
    Frontend->>Frontend: Re-enable input
    Frontend->>User: Display response with sources
```

## Component Responsibilities

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[User Interface<br/>script.js]
        UI --> |POST /api/query| API_LAYER
    end
    
    subgraph "API Layer" 
        API_LAYER[FastAPI App<br/>app.py]
        API_LAYER --> |orchestrate| RAG_LAYER
    end
    
    subgraph "RAG System Layer"
        RAG_LAYER[RAG System<br/>rag_system.py]
        RAG_LAYER --> SESSION[Session Manager]
        RAG_LAYER --> AI_GEN[AI Generator]
        RAG_LAYER --> TOOLS[Tool Manager]
    end
    
    subgraph "AI & Tools Layer"
        AI_GEN --> |API calls| CLAUDE[Anthropic Claude]
        TOOLS --> SEARCH[Course Search Tool]
        SEARCH --> |semantic search| VECTOR
    end
    
    subgraph "Data Layer"
        VECTOR[Vector Store<br/>ChromaDB]
        DOCS[Document Processor]
        DOCS --> |chunks| VECTOR
    end
    
    subgraph "External Services"
        CLAUDE
    end
```

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph "Input Processing"
        A[User Query] --> B[Session Context]
        B --> C[AI Prompt + Tools]
    end
    
    subgraph "Tool-Based Search"
        C --> D{Claude Decision}
        D -->|Use Tool| E[Course Search Tool]
        D -->|Direct Answer| H[Response Generation]
        E --> F[Vector Search]
        F --> G[Formatted Results]
        G --> H
    end
    
    subgraph "Response Assembly"
        H --> I[Source Collection]
        I --> J[Session Update]
        J --> K[JSON Response]
    end
    
    subgraph "Frontend Display"
        K --> L[Markdown Rendering]
        L --> M[Source Display]
        M --> N[User sees answer]
    end
```

## Key Design Patterns

1. **Tool-Based RAG**: AI decides when to search rather than always searching
2. **Session Persistence**: Conversation context maintained across queries
3. **Modular Architecture**: Clear separation of concerns
4. **Error Handling**: Comprehensive error handling at each layer
5. **Source Tracking**: UI transparency about information sources
6. **Async Processing**: Non-blocking API calls and database operations