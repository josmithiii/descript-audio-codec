```mermaid
graph TD
    graph LR
        A[Raw Audio] --> B[Volume Normalization<br>-16dB]
        B --> C[RescaleAudio<br>Ensure Max ≤ 1.0]
        C --> D[ShiftPhase<br>Optional]
        D --> E[To Model]

        style A fill:#f9f,stroke:#333
        style E fill:#f9f,stroke:#333

graph LR
    subgraph Input
        A[Audio Signal<br>Shape: 1 x T] 
    end

    subgraph "First WNConv1d Layer"
        B[Kernel Size: 7<br>Padding: 3<br>Weight Normalized]
    end

    subgraph Output
        C[Feature Maps<br>Shape: 64 x T]
    end

    A --> B --> C

    style A fill:#f9f,stroke:#333
    style C fill:#bbf,stroke:#333
```
