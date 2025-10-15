from a2a.types import Part, DataPart, TextPart, FilePart  # Only support official SDK part types


class PartsMessageHandler:
    """Handles construction, parsing, and validation of A2A Parts-based messages."""

    SUPPORTED_PART_TYPES = {
        "TextPart": TextPart,
        "DataPart": DataPart,
        "FilePart": FilePart,
    }


    @staticmethod
    def construct_part(part_type: str, content) -> Part:
        """Construct a Part object from type and content."""
        if part_type == "TextPart":
            return TextPart(text=content)
        elif part_type == "DataPart":
            return DataPart(data=content)
        elif part_type == "FilePart":
            if not isinstance(content, dict) or "file" not in content:
                raise ValueError("FilePart content must be a dict with a 'file' key")
            return FilePart(file=content["file"], **{k: v for k, v in content.items() if k != "file"})
        else:
            raise ValueError(f"Unsupported part type: {part_type}")

    @staticmethod
    def parse_part(part: Part) -> dict:
        """Parse a Part object into a serializable dict."""
        if isinstance(part, TextPart):
            return {"type": "TextPart", "content": part.text}
        elif isinstance(part, DataPart):
            return {"type": "DataPart", "content": part.data}
        elif isinstance(part, FilePart):
            return {"type": "FilePart", **part.__dict__}
        else:
            raise ValueError(f"Unsupported part instance: {type(part).__name__}")

    @staticmethod
    def validate_part(part: Part) -> bool:
        """Validate a Part object according to protocol rules."""
        if isinstance(part, TextPart):
            return isinstance(part.text, str) and len(part.text) > 0
        elif isinstance(part, DataPart):
            return part.data is not None
        elif isinstance(part, FilePart):
            return hasattr(part, "file") and part.file is not None
        else:
            return False 