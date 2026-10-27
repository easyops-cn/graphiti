# EasyOps Graphiti customizations

## Strict atomic revision storage types (2026-10-20)

FalkorDB considers stored Float `1.0` equal to Integer `1`. A numeric-only baseline comparison therefore admitted corrupted storage revisions even though the Action snapshot reader requires a positive integer. The atomic query now checks `typeOf(..._revision) = "Integer"` before comparing every edited entity/relationship, snapshot-only object and relation dependency baseline. All checks remain before the first write; rejected requests retain the entire graph, native IDs and receipts. There is no new configuration and no automatic legacy migration or change to legacy save paths.

### Upgrade notes

Preserve the type checks on all five baseline paths. `typeOf` was verified on the local modules 41607 and 41811; this function check does not remove the separate 41607 rollback/retry acceptance restriction. Real graph regression tests include valid integer controls and invalid missing, zero, negative, string, boolean and float storage values. Do not replace strict type checks with numeric coercion.

## Modification overview

| Area | Change | Scope |
|---|---|---|
| Atomic revision storage types | Reject numerically equal non-integer storage baselines in the atomic query | Edited objects, read snapshots and relation dependencies only |
| EntityNode.save / FalkorDB query | Optional atomic entity `_revision` maintenance | Bulk entity create/update only |
| EntityEdge.save / temporal replacement | Optional relation `_revision` maintenance and one-query replacement | Bulk relation create/update only |
| Typed entity bulk save | Default-off per-call revision option with one query and strict live receipt per entity | Internal Graphiti bulk helper only |
| Typed relation bulk save | Default-off per-call revision option with guarded one-query saves and strict live receipts | Internal Graphiti bulk helper only |
| Graphiti caller propagation | Keyword-only revision flags flow from triplet and episode APIs to entity/relation bulk saves | Graphiti API and selected service callers |
| Entity revision UUID pre-read | Untyped candidate lookup for cold `Entity` label and non-`Entity` UUID collisions | FalkorDB opted-in single and typed bulk entity queries only |
| Single relation cold type save | Conditional create and direct existing-relation update | FalkorDB opted-in `EntityEdge.save` only |
| Relation endpoint identity guard | Global endpoint UUID counts and physical node identity checks | FalkorDB opted-in single and typed bulk relation writers |
| Single-save effective no-op and receipt | Full persisted-state comparison, conditional write and checked live revision receipt | FalkorDB opted-in `EntityNode.save` and `EntityEdge.save` only |
| Entity write snapshot vectors | Explicit opt-in name/summary vector projection for full-replacement Bulk updates | `EntityNode.get_by_uuid(include_embeddings=True)` in Bulk entity write path |
| Bulk dedup tests | Align stale resolver-based tests with deterministic name/MinHash behavior | Tests only; no production dedup change |
| Structured Action graph write | Single-query, capability-gated entity/relation mutation with baseline checks and receipt | FalkorDB driver and shared Action graph-write boundary |
| Structured Action graph write | Guarded single-query FalkorDB executor with cross-kind UUID absence and null-safe snapshots | Shared Action graph-write boundary; explicit opt-in only |
| Action snapshot incarnation gate | Optional server-created `created_at` comparison for extant entity and edge baselines | FalkorDB atomic graph-write query only |
| Triplet relation metadata compatibility | Read-only `updated_at` storage metadata does not invalidate an otherwise identical Action edge snapshot | Atomic Action snapshot gate and reserved write properties |
| Isolated Entity delete gate | Ignore expired relations and Episode provenance when proving an Entity has no current business relations | FalkorDB atomic graph-write query only |
| Structured Triplet edge resolution | Filter same-endpoint dedup candidates by requested edge type and preserve explicit type/maturity before revision-aware save | `Graphiti.add_triplet` only; Episode extraction unchanged |

## Structured Triplet edge resolution

### Background

`Graphiti.add_triplet` previously passed all `RELATES_TO` edges between resolved endpoints to the generic edge resolver. Its duplicate fast path can reuse an edge of another logical type. The Triplet service then set `maturity_level` after the revision-aware save, bypassing the edge revision when the resolved UUID belonged to an existing edge. Generic LLM edge classification could also discard the explicit structured maturity value.

### Change and configuration

The Triplet path restricts duplicate candidates to the requested edge type and restores the explicitly requested type and `maturity_level` on the resolved edge before the existing typed bulk save. The service no longer performs a separate post-save maturity write. The existing guarded save initializes a new edge at revision 1, increments an existing edge only for an actual change, and leaves exact repeats unchanged. There is no new setting; Episode LLM extraction and Triplet's prior overwrite deletion remain unchanged.

### Upgrade notes

Keep the Triplet-only candidate filter and explicit metadata restoration when upgrading Graphiti. Verify a different-type edge on the same endpoints is not reused, and verify requested maturity survives generic edge resolution. Retain the existing physical-delete behavior of the Triplet overwrite path and the rule that edge changes do not increment endpoint revisions.

## Triplet relation metadata compatibility

### Background

The Triplet API stores an `updated_at` timestamp on business relations. The Action full-property snapshot gate previously treated that storage timestamp as an undeclared business field and rejected valid Actions that read a Triplet-created relation, even after a rejected lower-maturity submission left the relation unchanged.

### Change and configuration

The atomic edge snapshot gate recognizes `updated_at` as storage metadata. Structured write properties cannot set it. Business properties, relation identity, revision and temporal validity remain checked; an unknown business property still rejects the write. This applies only to the existing capability-gated Action path and requires no new configuration or change to Triplet deletion/overwrite behavior.

### Upgrade notes

Preserve both the snapshot metadata allowlist and the reserved-write restriction for `updated_at`. Keep the regression distinguishing this known timestamp from an unknown business property, and formally verify a rejected Triplet followed by execution of its existing Action baseline. This compatibility fix is not evidence that every Triplet direct-write path maintains revisions.

## Entity revision maintenance

### Background

Bulk entity upserts need a server-owned revision that reflects the live graph state. A revision computed from a previously loaded Python node can overwrite a concurrent update. Existing nodes may not yet have a revision.

### Change

`EntityNode.save(..., maintain_revision=True)` uses one FalkorDB save query that counts all nodes with the UUID and captures the stored revision before MERGE and property replacement. It permits mutation only when the UUID is absent or matches exactly one valid node; duplicate UUID matches fail before any property write, including when one match belongs to the current tenant. New nodes receive revision 1; changed existing nodes with a positive integer revision increment it; an unversioned existing node receives revision 2 on its first actual update. The incoming `_revision` attribute is excluded from the saved property map. Conflicting `uuid` or `group_id` attributes fail before graph I/O; matching metadata is rebound to the node's identity values before saving. Bulk request and direct-service inputs reject all three reserved properties. The query rejects cross-tenant UUID matches, invalid existing revisions, and tenantless existing nodes even when they carry revision zero. The save call requires exactly one returned record with the requested UUID. The default save path remains unchanged. Custom graph adapters and other providers reject the opt-in before writing.

The opted-in single-node query compares the complete effective replacement map except `_revision`, normalizing absent/null properties and converted vectors, including a null name vector, and also checks additive physical labels. The typed bulk-node query uses the same null-safe name-vector comparison while retaining its partial-property semantics. An exact replay performs no `SET` and retains its revision, including a legacy null revision. A real change increments once. The same-query return supplies the actual node UUID, stored revision and boolean change decision for strict internal validation; `EntityNode.save` continues to expose its prior UUID-only result row to callers.

`EntityNode.get_by_uuid(..., include_embeddings=True)` explicitly projects stored name and summary vectors into their model fields for Bulk entity updates that call the full-replacement single save query. The default read omits those fields, and the converter still removes vector keys from business `attributes` used by prompt/search consumers. Property-only updates and duplicate-alias-plus-property updates preserve both vectors; summary or new-alias updates preserve the name vector while replacing only the summary vector. The opt-in is a read-side write snapshot, not a new public API response field.

### Configuration

There is no environment setting. The Bulk entity service enables `maintain_revision=True` on its two actual save branches. Unchanged entities do not call save. Other graph write paths do not yet opt in.

## Relation revision maintenance

### Background

Bulk relation replacement previously expired the old edge and created the new edge in separate graph queries. A failed create could leave an expired old edge while the item reported failure. Relation attributes could also override stored revision and identity fields.

### Change

`EntityEdge.save(..., maintain_revision=True)` counts all relationships with the requested UUID, validates the exact source and target nodes and tenant, captures the live revision, then writes the relationship and revision in one FalkorDB query. A new edge starts at revision 1; an existing positive integer revision increments; a missing legacy revision uses baseline 1 and becomes 2. Duplicate UUIDs, malformed revisions, missing or duplicated endpoints, tenant mismatches, and mismatched identity or validity metadata reject the write. The default save path remains unchanged.

`EntityEdge.replace_temporally(driver, replacement)` counts the old UUID across all relationship types, requires one active same-tenant `RELATES_TO` edge with the expected logical type and endpoints, requires an unused replacement UUID and unique same-tenant endpoints, then expires the old edge and creates the replacement in one query. A same-named relationship with another physical type is rejected. The old revision increments once and the new UUID starts at 1. The helper checks the exact old/new UUID receipt. A failed item does not mark the Bulk graph as changed; successful siblings retain the existing partial-success behavior. Endpoint entity revisions are not changed.

For the opted-in single relation save, a cold `RELATES_TO` type exposed a separate defect: concurrent same-UUID writes could both return success while leaving two physical relationships. The guarded query now carries its untyped UUID match through endpoint validation, creates a typed relationship only when no UUID match exists, and updates the captured relationship directly otherwise. A same-query untyped post-write traversal returns the actual saved UUID only when exactly one matching `RELATES_TO` relationship exists. The default save and temporal replacement queries are unchanged.

The single relation query also counts endpoint UUID candidates across all node labels and requires exactly one `Entity` node per endpoint before a write. For an existing relation it compares the actual matched source and target nodes with those validated endpoints, not only their UUID strings. This closes the alias case where an `:Other` node shares an `:Entity` UUID and an existing relation attached to the `:Other` node could previously be mutated before receipt failure. The typed bulk relation query already uses untyped global endpoint counts and requires its globally matched UUID relation to equal the endpoint-bound existing relation; real source/target alias tests cover its create and existing cases. Self-loops remain valid and do not change endpoint revisions.

The opted-in single relation query compares the complete effective replacement map except `_revision`, including nullable temporal fields, episodes and the converted fact vector. Exact replay and null/absent-equivalent input skip all `SET` clauses, leaving even an unversioned legacy relation unchanged; a real change advances the stored revision once. A same-query post-write relationship read supplies actual UUID and revision plus the boolean change decision for strict internal validation. `EntityEdge.save` still returns a UUID-only row to callers; temporal replacement retains its separate old/new UUID receipt validator.

### Configuration

There is no environment setting. Bulk relation create calls the opt-in save; Bulk relation update calls temporal replacement after its unchanged check. Other Graphiti callers, typed bulk relation writers, Triplet, Episode, hard delete, and cascade delete retain their existing paths.

## Upgrade notes

This is lazy initialization, not a full migration of existing graph data. Do not treat `_revision` as a complete graph-wide concurrency gate until the remaining write paths and a migration are completed and validated. Bulk responses and partial-success behavior remain unchanged. A future Graphiti upgrade must retain both FalkorDB query paths' global UUID counts, endpoint and tenant gates, live revision capture, identity validation before graph I/O, and exact return shapes. Opted-in entity queries must use an untyped UUID pre-read followed by a runtime `Entity` label check, including on a graph where the label has never existed. The Bulk full-replacement update path must request both stored vectors explicitly; default entity reads must keep vectors out of model fields and prompt attributes unless requested. Opted-in relation queries must count all endpoint UUID candidates across labels and compare existing relation endpoint node identity before writes; UUID string equality alone is insufficient. Opted-in single saves must compare the effective full property map (and node labels), skip all writes for exact no-ops, and validate live UUID/revision/changed receipts while preserving UUID-only external rows. Opted-in single relation saves must not unconditionally typed-`MERGE` an existing UUID on a cold physical type; keep conditional creation, direct updates and a post-write actual-UUID receipt. Temporal replacement must remain a single query so failure does not leave only the old edge expired.

## Typed entity bulk revision option

## Structured Action graph write

### Extant snapshot incarnation baseline

The shared strict snapshot model accepts optional, timezone-aware `created_at` metadata from trusted graph reads. Action entity and edge snapshots preserve it separately from business properties, and signed plans carry it into the shared expected state. The FalkorDB atomic query compares an asserted value against the live entity or edge `created_at` in the same pre-mutation gate as revision and property checks. A hard delete followed by same-UUID recreation with matching business state and revision therefore cannot satisfy a baseline bearing the old creation time. A missing value retains the existing ordinary-write behavior; it is not proof of incarnation for revert.

There is no configuration setting. This comparison applies only when the server-authored baseline supplies `created_at`; business edits still cannot set creation metadata. During Graphiti upgrades, preserve both entity and edge comparisons inside the atomic gate, the optional legacy behavior, and the server-owned creation assignment. Do not treat this as deletion-tombstone or expired-relation-lineage protection, and do not enable the disabled executor on its strength alone.

`FalkorDriver.execute_atomic_graph_write()` is the explicit Graphiti capability used by the Action graph-write boundary. It accepts a validated `GraphWriteRequest` and trusted tenant context, constructs one parameterized Cypher query, and invokes FalkorDB once. The query collects untyped UUID candidates, checks tenant and `Entity` identity, revisions, endpoint uniqueness, active relation selection, UUID absence, isolated entity constraints, relation dependency results (including an explicit empty result), and complete entity or edge snapshots before entering gated `FOREACH` mutations. Entity create/update/delete and relation create/update/temporal-expire are supported; hard relation deletion remains explicitly rejected.

Logical types are validated before any identifier interpolation. Business properties are JSON serializable and cannot overwrite graph identity, revision, temporal, or Action provenance fields. Creates initialize `_revision` to `1`; updates and temporal expiration increment the live revision and set `_action_execution_id` and `_updated_at`. The receipt includes request and tenant identity, the gate result, outcome, and planned change count. Drivers without this capability remain fail-closed, and legacy Graphiti save paths do not opt in.

This initial executor is disabled by default: `execute_atomic_graph_write()` rejects requests unless the internal `_atomic_graph_write_verified` test gate is explicitly enabled. No production constructor enables it. The query stores an `ActionGraphWriteReceipt` in the same gated query, returning `REPLAYED` for an identical request/hash and `GRAPH_WRITE_HASH_CONFLICT` for a different hash. Relation dependencies are re-resolved against active endpoint-bound edges; snapshots compare all business keys with null/absent equivalence and reject unexpected non-null keys. The older raw-query atomicity experiment is not acceptance evidence for this generated executor. Preserve the default rejection during upgrades until deployment, authorization-version and recovery validation are completed.

The executor targets atomicity for the FalkorDB graph query only. PostgreSQL execution records and permission policy checks remain service-layer concerns and are not claimed to share this transaction.

Generated-query regression checks run against a private FalkorDB graph, rather than the older handwritten atomicity experiment. Absence assertions collect both node and edge candidates for every asserted UUID. Existing relation endpoints require exactly one global node candidate before tenant and Entity-label validation. Edge snapshot temporal comparisons treat two absent values as equal. Upgrades must retain these gates: a tenant-filtered endpoint count alone admits duplicate UUID aliases, and direct equality of null temporal fields falsely rejects active edges. These driver integration checks are not formal API or UI acceptance and do not enable the production gate.

Creates also initialize Graphiti's required storage metadata: entity name, summary and creation time; relationship fact, episodes and creation time. Supplied business names and summaries are retained. Snapshot key filtering excludes storage metadata removed by Graphiti model readers, while entity name remains part of the business baseline. A persisted replay receipt must match the trusted person, source reference and APPLIED state as well as the request hash; incomplete legacy receipts fail closed. This does not establish exactly-once external effects, which require the service's durable delivery protocol.

The same global endpoint identity gate applies to edited relations, relation read dependencies and edge snapshots. Snapshot-only reads must honor a separately supplied revision even when the snapshot itself has no revision. Creation timestamps, episode provenance, embedding vectors and physical endpoint fields cannot be supplied as business edits. Preserve these restrictions when upgrading: they prevent metadata overwrite and snapshot-based bypass of concurrency checks.

Isolated Entity deletion checks current business relations only. The atomic query requires every incident relation used by the isolation gate to be active and to connect two `Entity` nodes. Expired or invalidated relation history remains in the graph for traceability but does not block deletion, and Episode `MENTIONS` provenance is not a business relation. Active relations between Entities still reject deletion. This keeps Action deletion consistent with the Function and ontology-tool guards while preserving temporal history.

Action temporal replacement is distinct from ordinary relation creation. The internal request carries a timezone-aware `effective_at` and a create edit's `previous_edge_uuid`, validated to reference a same-request expiration of the same type and endpoints with a positive version baseline. The single query expires the old edge and creates the new edge at that signed plan time, preserving lineage and inheriting old revision plus one. Ordinary edge creation still starts at one; Bulk's existing replacement behavior is unchanged. Preserve this structured metadata validation and never accept lineage or revision through business properties.

Compiler-resolved read-only objects also enter the graph request's snapshots and revision gates; they cannot be omitted merely because the Action does not edit them. Entity name is Graphiti storage metadata rather than a Schema property, so it is ignored when absent from a business snapshot; if explicitly included in a snapshot, it is still compared. This keeps schema-only Action snapshots compatible with formal entity reads without dropping their declared business keys.

### Background

Graphiti's typed bulk writer saves each entity once per label and does not receive query results through its standard FalkorDB session `run` method. A revision increment in the old loop would increment one object multiple times and could silently accept a zero-row guarded query.

### Change

The internal `add_nodes_and_edges_bulk(..., maintain_entity_revisions=True)` option uses one guarded FalkorDB query per entity and checks an exact UUID, revision, and boolean change receipt from the same query result. The query compares persisted fields, requested labels, and embeddings before deciding whether to save. It reads the current graph revision before any change, initializes new entities at 1, increments changed entities, and leaves unchanged entities untouched. A positive integer revision is required for new or changed rows; an unchanged legacy row may return a null revision. Attribute-provided `labels` must match the trusted node labels plus `Entity` as a set; conflicting metadata fails before graph I/O, and matching metadata is rebound to the trusted labels after attribute sanitization. The default query path stays unchanged, including its per-label query loop.

### Configuration

This is a call-scoped Python keyword option, defaulting to false. It is not an environment setting or client API parameter. Triplet, Episode, typed relations, and other callers do not enable it in this slice.

### Upgrade notes

Keep the per-entity query count, no-op detection, exact live UUID/revision/changed receipts, trusted-label validation, and default-off behavior when upgrading Graphiti. The default path must still execute every per-label query; never pair that query list with nodes using `zip`. This option does not make the surrounding multi-query ingestion atomic and does not complete revision coverage for typed relations or other entry points.

## Cold Entity label UUID guard

### Background

In isolated FalkorDB testing, two concurrent first-`Entity` saves for one UUID and different properties both returned success while the only node remained at revision 1. This occurred in both the opted-in single save and typed bulk writer; priming the `Entity` label elsewhere made both reach revision 2. The former `OPTIONAL MATCH (candidate:Entity {uuid})` also missed a non-`Entity` node with the same UUID. The deployed planner's exact scheduling cause is not established.

### Change

Both opted-in node queries now find UUID candidates without a label in the pre-write traversal. The guard counts all matching nodes and accepts only zero matches or exactly one same-tenant candidate carrying the `Entity` label and a valid stored revision. A non-`Entity` UUID collision, duplicate UUID or foreign-tenant candidate yields no success receipt before `MERGE` or property writes. Creation and existing-node revision updates remain in the same query. The `MERGE (n:Entity {uuid})`, single-save full replacement, typed bulk partial-property/no-op behavior, labels, vectors and default save paths are unchanged.

### Configuration

No new setting or caller flag. This applies only when `maintain_revision=True` or `maintain_entity_revisions=True` is already selected.

### Upgrade notes

Retain the untyped global UUID pre-read and runtime `Entity` label guard for both opted-in node queries. A type-scoped pre-read can lose the first-label increment and cannot reject a non-`Entity` UUID collision. The isolated tests cover cold and primed concurrent saves plus collision handling, but do not establish Graphiti-wide caller or API acceptance.

## Typed relation bulk revision option

### Background

The default typed relation bulk writer batches by logical edge name and uses endpoint/type `MERGE`, so later input for the same endpoints and physical type replaces earlier properties and UUID. It does not maintain revisions, and attribute metadata can overwrite trusted identity fields.

### Change

`add_nodes_and_edges_bulk(..., maintain_edge_revisions=True)` keeps the same grouped input order but executes one guarded FalkorDB query per relation. The query counts endpoint UUIDs across all node labels and all incoming-UUID relationship matches, requires exactly one `Entity` per endpoint, rejects ambiguous or cross-tenant matches, validates the existing physical/logical type and live revision, compares the complete persisted property set except `_revision` with null-removal and vector semantics, then conditionally writes. Its endpoint-bound existing-relation read matches any physical relationship and filters `type(typed)` against the validated logical name at runtime; only the create branch embeds the physical type in the query. An absent relation or replacement UUID starts at revision 1; a changed same-UUID relation increments the live revision, with a legacy missing revision advancing to 2. After the conditional write, the query matches the saved endpoint-bound relationship and returns its actual UUID and stored revision together with the boolean change decision; an unchanged legacy relation may therefore return a null revision. An identical relation receives a no-op receipt without a write. Entity endpoint revisions are never modified. Identity metadata supplied in attributes must agree with the edge object and is rebound after sanitization; incoming `_revision` is ignored. The default batched `UNWIND` writer is unchanged.

### Configuration

This internal Python keyword option defaults to false. It requires FalkorDB and its result-bearing session, and is not enabled by Triplet, Episode, HTTP, MCP, or another service in this slice. It is not an environment setting.

### Upgrade notes

Keep the default batch and last-input-wins behavior, exact live UUID/revision/changed/previous-UUID receipt checking, global UUID and endpoint cardinality guards, full-map no-op comparison, runtime physical-type filtering on the read, and call-scoped opt-in. Each edge query is atomic, but the surrounding bulk ingestion runs multiple independent queries; it is not a transaction across a batch. The former typed `OPTIONAL MATCH` returned duplicate relationships or zero rows for concurrent first-type writes in isolated FalkorDB testing. The one-clause runtime-filter repair passed six isolated concurrency cases and the 25-case typed relation real suite, including a different-physical-type sentinel; independent broad regression and caller-level acceptance remain pending, so no caller enables this option. Typed relation coverage does not complete graph-wide revision consistency.

## Graphiti caller propagation

The public `Graphiti.add_triplet`, `Graphiti.add_episode`, and
`Graphiti.add_episode_bulk` methods accept keyword-only
`maintain_entity_revisions=False` and `maintain_edge_revisions=False` options. The
flags are passed only to entity and business-relation bulk saves. Episodic nodes
and `EpisodicEdge` audit links never receive these flags. `add_episode_bulk` keeps
its first Episode-only save on the default path and applies the options only to the
second extracted entity/relation save. The options are call-scoped and are not
stored on cached Graphiti clients.

The Triplet service enables both options for its Graphiti call. The synchronous
Episode service enables both options for extracted entity/relation persistence,
including enrichment of an existing episode; the fast EpisodicNode save remains
unversioned. The service-level changes preserve existing response and partial
success behavior. Direct maturity, deletion, audit Episode/MENTIONS writes, and
other non-Graphiti paths remain separate revision work.

## Bulk dedup test baseline

The in-memory `dedupe_nodes_bulk` path uses deterministic normalized-name and MinHash matching rather than the former `resolve_extracted_nodes` callback. Three stale callback-based tests were repaired to assert real canonical object reuse, directed UUID maps, sparse singleton/unmatched maps, and that the external resolver is not called; a real MinHash case was added. Production deduplication was not changed. The parent regression run passed 676 tests with two existing warnings. Future upgrades should preserve tests against the algorithm that actually executes rather than mocked legacy callbacks. The current batch-wide candidate search does not enforce tenant or label boundaries; that guarantee is not asserted by these repaired tests.
