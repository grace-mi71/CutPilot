# frontend — deferred

Intentionally empty until **Week 9–10**.

Building the UI now would leave it stale while the backend pipeline is still
moving. What *is* fixed early is the **data contract** the UI consumes, defined
in `backend/cutpilot/schemas/`:

| Schema       | What the UI does with it                                  |
|--------------|-----------------------------------------------------------|
| `VideoAsset` | proxy video for the player, sprite sheet for the filmstrip |
| `Track`      | per-object presence bars on the timeline                   |
| `EditPlan`   | operation bars on the timeline + the editable plan panel   |

A throwaway single-file prototype (static HTML + one canvas timeline) is planned
around Week 5–6 to de-risk timeline rendering before the real build starts.

See `docs/decisions/0001-initial-stack.md` for why this is deferred.
