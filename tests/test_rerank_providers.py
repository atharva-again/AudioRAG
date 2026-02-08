"""Unit tests for reranker providers - instantiation tests."""



class TestRerankerProvidersInstantiation:
    """Test reranker provider instantiation through lazy imports."""

    def test_passthrough_reranker_instantiation(self) -> None:
        """Test Passthrough reranker can be instantiated."""
        from audiorag.rerank import PassthroughReranker

        reranker = PassthroughReranker()
        assert reranker is not None
