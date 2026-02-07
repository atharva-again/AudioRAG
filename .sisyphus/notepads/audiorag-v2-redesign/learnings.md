## Protocol Conformance Testing Pattern

Created comprehensive protocol conformance tests in `tests/test_protocol_conformance.py`:

### Test Structure
- **Per-protocol test classes**: Each protocol gets its own test class (e.g., `TestSTTProviderConformance`)
- **Fake implementations**: Use simple fake classes (not mocks) to verify protocol compliance
- **Contract verification**: Tests verify method signatures and return type shapes, not implementation details

### Key Testing Patterns
1. **Runtime checkable verification**: Verify `_is_protocol` attribute exists
2. **Method presence checks**: Verify required methods exist using `hasattr()`
3. **isinstance() validation**: Verify compliant implementations pass `isinstance()` checks
4. **Return type shape testing**: Verify return values have correct structure (list, dict, etc.)
5. **Non-compliant detection**: Verify missing/wrong methods fail `isinstance()` checks

### Important Discoveries
- **runtime_checkable limitation**: Only checks method presence, NOT signatures or return types
- **Fake > Mock**: Simple fake classes are clearer than AsyncMock for protocol testing
- **Shape over values**: Test return type structure, not specific values (contract tests)
- **Ruff noqa needed**: Use `# ruff: noqa: ARG002, PLC0415` for unused args in fake implementations

### ChunkingStrategy Protocol (Future)
- Added skipped tests for ChunkingStrategy protocol (not yet implemented)
- Expected signature: `chunk(segments: list[TranscriptionSegment], source_url: str, title: str) -> list[ChunkMetadata]`
- Tests will auto-enable when protocol is implemented

### Integration Test Pattern
- `TestProtocolInteroperability` demonstrates full pipeline with all fake providers
- Verifies protocols work together end-to-end
- No external dependencies or API keys required

### Coverage
- 36 passing tests covering all 6 existing protocols
- 5 skipped tests for ChunkingStrategy (future protocol)
- All tests pass without external API keys or network calls
