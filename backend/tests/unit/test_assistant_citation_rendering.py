from app.services.assistant_gateway_service import AssistantGatewayService


def test_dedupe_references_preserves_numbered_internal_refs_and_leaves_web_refs_unnumbered():
    service = AssistantGatewayService.__new__(AssistantGatewayService)

    references = service._dedupe_references(
        [
            {
                "type": "internal",
                "title": "Feature run snapshot",
                "url": "/stocks/NVDA",
                "reference_number": 7,
            },
            {
                "type": "web",
                "title": "Reuters",
                "url": "https://example.com",
                "reference_number": None,
            },
        ]
    )

    assert references == [
        {
            "type": "internal",
            "title": "Feature run snapshot",
            "url": "/stocks/NVDA",
            "reference_number": 1,
        },
        {
            "type": "web",
            "title": "Reuters",
            "url": "https://example.com",
            "reference_number": None,
        },
    ]
