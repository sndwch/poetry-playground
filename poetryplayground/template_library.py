"""Template library for managing and persisting poem templates.

This module provides the TemplateLibrary class for storing, loading, and
managing PoemTemplate instances. Templates can be persisted to disk as JSON
files and loaded on demand.

The library includes built-in templates for common forms (haiku, tanka) and
supports user-defined custom templates.

Example:
    >>> from poetryplayground.template_library import get_template_library
    >>> library = get_template_library()
    >>> haiku = library.get_template("haiku")
    >>> library.add_template(my_custom_template)
    >>> library.save_template("my_template")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from poetryplayground.core.quality_scorer import EmotionalTone
from poetryplayground.logger import logger
from poetryplayground.poem_template import (
    PoemTemplate,
    create_haiku_template,
    create_tanka_template,
)


class TemplateLibrary:
    """Library for managing poem templates.

    Manages a collection of PoemTemplates with persistence to disk.
    Templates are stored as JSON files in ~/.poetryplayground/templates/

    Attributes:
        templates_dir: Directory containing template JSON files
        templates: Dictionary mapping template names to PoemTemplate objects
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize template library.

        Args:
            templates_dir: Directory for storing templates
                          (default: ~/.poetryplayground/templates)
        """
        if templates_dir is None:
            templates_dir = Path.home() / ".poetryplayground" / "templates"

        self.templates_dir = templates_dir
        self.templates: Dict[str, PoemTemplate] = {}

        # Create templates directory if it doesn't exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Load built-in templates
        self._load_builtin_templates()

        # Load user templates from disk
        self._load_user_templates()

        logger.info(f"TemplateLibrary initialized with {len(self.templates)} templates")

    def _load_builtin_templates(self) -> None:
        """Load built-in default templates."""
        # Haiku template
        haiku = create_haiku_template()
        self.templates["haiku"] = haiku

        # Dark haiku variant
        dark_haiku = create_haiku_template(
            semantic_domains=["darkness", "night", "shadow"],
            emotional_tone=EmotionalTone.DARK,
        )
        dark_haiku.title = "Dark Haiku Template"
        self.templates["haiku_dark"] = dark_haiku

        # Nature haiku variant
        nature_haiku = create_haiku_template(
            semantic_domains=["nature", "seasons", "weather"],
            emotional_tone=EmotionalTone.NEUTRAL,
        )
        nature_haiku.title = "Nature Haiku Template"
        self.templates["haiku_nature"] = nature_haiku

        # Tanka template
        tanka = create_tanka_template()
        self.templates["tanka"] = tanka

        # Emotional tanka variant
        emotional_tanka = create_tanka_template(
            semantic_domains=["emotion", "love", "longing"],
            emotional_tone=EmotionalTone.MIXED,
        )
        emotional_tanka.title = "Emotional Tanka Template"
        self.templates["tanka_emotional"] = emotional_tanka

        logger.debug(f"Loaded {len(self.templates)} built-in templates")

    def _load_user_templates(self) -> None:
        """Load user templates from disk."""
        loaded_count = 0

        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, encoding="utf-8") as f:
                    data = json.load(f)
                    template = PoemTemplate.from_dict(data)
                    # Use filename (without .json) as template name
                    template_name = template_file.stem
                    self.templates[template_name] = template
                    loaded_count += 1
            except Exception as e:
                logger.error(f"Error loading template from {template_file}: {e}")

        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} user templates from {self.templates_dir}")

    def get_template(self, name: str) -> Optional[PoemTemplate]:
        """Get template by name.

        Args:
            name: Template name (e.g., "haiku", "tanka", "my_template")

        Returns:
            PoemTemplate if found, None otherwise
        """
        return self.templates.get(name)

    def add_template(self, template: PoemTemplate, name: Optional[str] = None) -> str:
        """Add template to library.

        Args:
            template: PoemTemplate to add
            name: Optional name for the template. If None, uses sanitized title

        Returns:
            The name used to store the template
        """
        if name is None:
            # Generate name from title (lowercase, replace spaces with underscores)
            name = template.title.lower().replace(" ", "_")

        # Sanitize name (alphanumeric and underscores only)
        name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)

        self.templates[name] = template
        logger.info(f"Added template '{name}' to library")
        return name

    def remove_template(self, name: str) -> bool:
        """Remove template from library.

        Args:
            name: Template name to remove

        Returns:
            True if template was removed, False if not found

        Note:
            Built-in templates (haiku, tanka, etc.) can be removed from memory
            but will be reloaded on next initialization.
        """
        if name in self.templates:
            del self.templates[name]
            logger.info(f"Removed template '{name}' from library")
            return True
        return False

    def save_template(self, name: str, overwrite: bool = False) -> bool:
        """Save template to disk.

        Args:
            name: Name of template to save
            overwrite: If True, overwrite existing file

        Returns:
            True if saved successfully, False otherwise

        Raises:
            FileExistsError: If file exists and overwrite=False
        """
        if name not in self.templates:
            logger.error(f"Template '{name}' not found in library")
            return False

        template_file = self.templates_dir / f"{name}.json"

        if template_file.exists() and not overwrite:
            raise FileExistsError(
                f"Template file {template_file} already exists. Use overwrite=True to replace it."
            )

        try:
            template = self.templates[name]
            data = template.to_dict()

            with open(template_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved template '{name}' to {template_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving template '{name}': {e}")
            return False

    def load_template(self, filepath: Path) -> Optional[str]:
        """Load template from external JSON file.

        Args:
            filepath: Path to JSON file containing template

        Returns:
            Template name if loaded successfully, None otherwise
        """
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                template = PoemTemplate.from_dict(data)

            # Use filename (without .json) as template name
            name = filepath.stem
            self.templates[name] = template

            logger.info(f"Loaded template '{name}' from {filepath}")
            return name
        except Exception as e:
            logger.error(f"Error loading template from {filepath}: {e}")
            return None

    def list_templates(self) -> List[str]:
        """Get list of all template names.

        Returns:
            Sorted list of template names
        """
        return sorted(self.templates.keys())

    def search_templates(
        self,
        lines: Optional[int] = None,
        syllable_pattern: Optional[List[int]] = None,
        semantic_domain: Optional[str] = None,
        emotional_tone: Optional[EmotionalTone] = None,
    ) -> List[str]:
        """Search for templates matching criteria.

        Args:
            lines: Filter by number of lines
            syllable_pattern: Filter by exact syllable pattern
            semantic_domain: Filter by semantic domain (must include this domain)
            emotional_tone: Filter by emotional tone

        Returns:
            List of template names matching all specified criteria
        """
        matches = []

        for name, template in self.templates.items():
            # Check line count
            if lines is not None and template.lines != lines:
                continue

            # Check syllable pattern
            if syllable_pattern is not None and not template.matches_structure(syllable_pattern):
                continue

            # Check semantic domain
            if semantic_domain is not None and semantic_domain not in template.semantic_domains:
                continue

            # Check emotional tone
            if emotional_tone is not None and template.emotional_tone != emotional_tone:
                continue

            matches.append(name)

        return sorted(matches)

    def get_templates_by_form(self, form_type: str) -> List[str]:
        """Get templates for a specific poetic form.

        Args:
            form_type: Form type (e.g., "haiku", "tanka", "sonnet")

        Returns:
            List of template names matching the form type
        """
        form_type_lower = form_type.lower()
        return [
            name
            for name in self.templates
            if form_type_lower in name.lower()
            or form_type_lower in self.templates[name].title.lower()
        ]

    def export_template(self, name: str, filepath: Path) -> bool:
        """Export template to external JSON file.

        Args:
            name: Name of template to export
            filepath: Destination file path

        Returns:
            True if exported successfully, False otherwise
        """
        if name not in self.templates:
            logger.error(f"Template '{name}' not found in library")
            return False

        try:
            template = self.templates[name]
            data = template.to_dict()

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported template '{name}' to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting template '{name}': {e}")
            return False

    def get_template_info(self, name: str) -> Optional[Dict]:
        """Get summary information about a template.

        Args:
            name: Template name

        Returns:
            Dictionary with template summary info, or None if not found
        """
        if name not in self.templates:
            return None

        template = self.templates[name]
        return {
            "name": name,
            "title": template.title,
            "source": template.source,
            "author": template.author,
            "lines": template.lines,
            "total_syllables": template.get_total_syllables(),
            "syllable_pattern": template.syllable_pattern,
            "semantic_domains": template.semantic_domains,
            "metaphor_types": template.metaphor_types,
            "emotional_tone": template.emotional_tone.value,
            "formality_level": template.formality_level.value,
            "has_line_templates": len(template.line_templates) > 0,
        }

    def __len__(self) -> int:
        """Return number of templates in library."""
        return len(self.templates)

    def __contains__(self, name: str) -> bool:
        """Check if template exists in library."""
        return name in self.templates

    def __str__(self) -> str:
        """Human-readable library summary."""
        return f"TemplateLibrary({len(self.templates)} templates)"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"TemplateLibrary(templates_dir={self.templates_dir}, count={len(self.templates)})"


# Global library instance
_library_instance: Optional[TemplateLibrary] = None


def get_template_library(templates_dir: Optional[Path] = None) -> TemplateLibrary:
    """Get global TemplateLibrary instance (singleton pattern).

    Args:
        templates_dir: Optional custom templates directory

    Returns:
        Global TemplateLibrary instance
    """
    global _library_instance
    if _library_instance is None or templates_dir is not None:
        _library_instance = TemplateLibrary(templates_dir)
    return _library_instance
