"""
Unit tests for MentionExtractor.
"""

import pytest
from src.core.extractors import MentionExtractor


class TestMentionExtractor:
    """Test suite for MentionExtractor."""

    def test_single_mention(self):
        """Test extraction of a single mention."""
        extractor = MentionExtractor()
        text = "Hey @alice, how are you?"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == '@alice'
        assert result[0]['type'] == 'MENTION'
        assert result[0]['score'] == 1.0

    def test_multiple_mentions(self):
        """Test extraction of multiple mentions."""
        extractor = MentionExtractor()
        text = "Thanks @alice and @bob for the help! @charlie"
        result = extractor.extract_from_text(text)

        assert len(result) == 3
        mentions = [r['text'] for r in result]
        assert '@alice' in mentions
        assert '@bob' in mentions
        assert '@charlie' in mentions

    def test_case_normalization(self):
        """Test case normalization."""
        extractor = MentionExtractor(normalize_case=True)
        text = "@Alice @ALICE @alice"
        result = extractor.extract_from_text(text)

        # Should deduplicate to single lowercase version
        assert len(result) == 1
        assert result[0]['text'] == '@alice'

    def test_case_preservation(self):
        """Test case preservation when normalization is off."""
        extractor = MentionExtractor(normalize_case=False)
        text = "@Alice @ALICE @alice"
        result = extractor.extract_from_text(text)

        # Should keep all three
        assert len(result) == 3
        mentions = [r['text'] for r in result]
        assert '@Alice' in mentions
        assert '@ALICE' in mentions
        assert '@alice' in mentions

    def test_exclude_emails(self):
        """Test that email addresses are excluded."""
        extractor = MentionExtractor(exclude_emails=True)
        text = "Contact me at user@example.com or @alice on Twitter"
        result = extractor.extract_from_text(text)

        # Should only get @alice, not the email
        assert len(result) == 1
        assert result[0]['text'] == '@alice'

    def test_include_emails(self):
        """Test that emails can be included if requested."""
        extractor = MentionExtractor(exclude_emails=False)
        text = "Contact @alice or reach me at user@example.com"
        result = extractor.extract_from_text(text)

        # Should get both
        assert len(result) >= 1
        mentions = [r['text'] for r in result]
        assert '@alice' in mentions

    def test_empty_text(self):
        """Test handling of empty text."""
        extractor = MentionExtractor()
        result = extractor.extract_from_text("")
        assert result == []

    def test_none_text(self):
        """Test handling of None text."""
        extractor = MentionExtractor()
        result = extractor.extract_from_text(None)
        assert result == []

    def test_no_mentions(self):
        """Test text with no mentions."""
        extractor = MentionExtractor()
        text = "This text has no mentions at all."
        result = extractor.extract_from_text(text)
        assert result == []

    def test_duplicate_mentions_in_same_text(self):
        """Test that duplicate mentions in same text are deduplicated."""
        extractor = MentionExtractor()
        text = "Thanks @alice! Yes, @alice is great! @alice rocks!"
        result = extractor.extract_from_text(text)

        # Should only return one instance
        assert len(result) == 1
        assert result[0]['text'] == '@alice'

    def test_mention_with_numbers(self):
        """Test mentions containing numbers."""
        extractor = MentionExtractor()
        text = "Follow @user123 and @alice2023 for updates"
        result = extractor.extract_from_text(text)

        assert len(result) == 2
        mentions = [r['text'] for r in result]
        assert '@user123' in mentions
        assert '@alice2023' in mentions

    def test_unicode_mentions(self):
        """Test extraction of Unicode mentions."""
        extractor = MentionExtractor()
        text = "Check out @用户 and @utilisateur for more"
        result = extractor.extract_from_text(text)

        assert len(result) == 2
        mentions = [r['text'] for r in result]
        assert '@用户' in mentions
        assert '@utilisateur' in mentions

    def test_batch_extraction(self):
        """Test batch extraction."""
        extractor = MentionExtractor()
        texts = [
            "Thanks @alice",
            "Hey @bob and @charlie",
            "No mentions here"
        ]
        results = extractor.extract_batch(texts)

        assert len(results) == 3
        assert len(results[0]) == 1
        assert results[0][0]['text'] == '@alice'
        assert len(results[1]) == 2
        assert len(results[2]) == 0

    def test_get_extractor_type(self):
        """Test extractor type identifier."""
        extractor = MentionExtractor()
        assert extractor.get_extractor_type() == 'mention'

    def test_get_config(self):
        """Test configuration retrieval."""
        extractor = MentionExtractor(normalize_case=False, exclude_emails=False)
        config = extractor.get_config()

        assert 'normalize_case' in config
        assert config['normalize_case'] is False
        assert 'exclude_emails' in config
        assert config['exclude_emails'] is False

    def test_mention_at_start(self):
        """Test mention at the start of text."""
        extractor = MentionExtractor()
        text = "@alice is amazing"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == '@alice'

    def test_mention_at_end(self):
        """Test mention at the end of text."""
        extractor = MentionExtractor()
        text = "Thanks for the help @alice"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == '@alice'

    def test_mention_with_underscores(self):
        """Test mentions with underscores."""
        extractor = MentionExtractor()
        text = "Follow @user_name and @another_user"
        result = extractor.extract_from_text(text)

        assert len(result) == 2
        mentions = [r['text'] for r in result]
        assert '@user_name' in mentions
        assert '@another_user' in mentions

    def test_email_detection_heuristic(self):
        """Test that the email detection heuristic works."""
        extractor = MentionExtractor(exclude_emails=True)

        # This should be filtered as an email
        text1 = "Email me at john@company.com"
        result1 = extractor.extract_from_text(text1)
        assert len(result1) == 0  # Should filter out the email

        # This should be kept as a mention
        text2 = "Follow @john on Twitter"
        result2 = extractor.extract_from_text(text2)
        assert len(result2) == 1
        assert result2[0]['text'] == '@john'
