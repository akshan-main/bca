"""Shared test fixtures for CeGraph."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with sample Python files."""
    # Main module
    (tmp_path / "main.py").write_text('''"""Main application entry point."""

from utils import helper_function, calculate_total
from models import User, Order


def main():
    """Run the main application."""
    user = User("Alice", "alice@example.com")
    order = Order(user, items=["widget", "gadget"])
    total = calculate_total(order.items)
    result = helper_function(total)
    print(f"Order total: {result}")
    return result


def parse_arguments():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main()
''')

    # Utils module
    (tmp_path / "utils.py").write_text('''"""Utility functions."""

TAX_RATE = 0.08


def helper_function(value):
    """Apply formatting to a value."""
    return f"${value:.2f}"


def calculate_total(items):
    """Calculate total price for a list of items."""
    prices = {"widget": 9.99, "gadget": 24.99, "doohickey": 4.99}
    subtotal = sum(prices.get(item, 0) for item in items)
    tax = subtotal * TAX_RATE
    return subtotal + tax


def validate_email(email):
    """Validate an email address."""
    import re
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))
''')

    # Models module
    (tmp_path / "models.py").write_text('''"""Data models."""


class User:
    """Represents a user in the system."""

    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def display_name(self):
        """Get the display name."""
        return self.name.title()

    def is_valid(self):
        """Check if user data is valid."""
        from utils import validate_email
        return bool(self.name) and validate_email(self.email)


class Order:
    """Represents an order."""

    def __init__(self, user: User, items: list):
        self.user = user
        self.items = items

    def get_total(self):
        """Get the order total."""
        from utils import calculate_total
        return calculate_total(self.items)

    def summary(self):
        """Get order summary string."""
        total = self.get_total()
        return f"Order for {self.user.display_name()}: {len(self.items)} items, ${total:.2f}"
''')

    # A subdirectory with more files
    api_dir = tmp_path / "api"
    api_dir.mkdir()

    (api_dir / "__init__.py").write_text('"""API package."""\n')

    (api_dir / "routes.py").write_text('''"""API routes."""

from models import User, Order


def get_user(user_id):
    """Get a user by ID."""
    # Simulated database lookup
    return User("Test User", "test@example.com")


def create_order(user_id, items):
    """Create a new order."""
    user = get_user(user_id)
    order = Order(user, items)
    return {"total": order.get_total(), "summary": order.summary()}


def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
''')

    return tmp_path


@pytest.fixture
def sample_python_source() -> str:
    """Sample Python source code for parser testing."""
    return '''"""Sample module."""

import os
from typing import List, Optional
from pathlib import Path


CONSTANT_VALUE = 42


class BaseProcessor:
    """Base class for processors."""

    def __init__(self, name: str):
        self.name = name

    def process(self, data: List[str]) -> List[str]:
        """Process the data."""
        return [self._transform(item) for item in data]

    def _transform(self, item: str) -> str:
        """Transform a single item."""
        return item.strip()


class AdvancedProcessor(BaseProcessor):
    """Advanced processor with extra features."""

    def __init__(self, name: str, verbose: bool = False):
        super().__init__(name)
        self.verbose = verbose

    def process(self, data: List[str]) -> List[str]:
        """Process with logging."""
        if self.verbose:
            print(f"Processing {len(data)} items")
        return super().process(data)

    def batch_process(self, batches: List[List[str]]) -> List[List[str]]:
        """Process multiple batches."""
        return [self.process(batch) for batch in batches]


def create_processor(name: str, advanced: bool = False) -> BaseProcessor:
    """Factory function for creating processors."""
    if advanced:
        return AdvancedProcessor(name, verbose=True)
    return BaseProcessor(name)


def run_pipeline(items: List[str], processor_name: str = "default") -> List[str]:
    """Run the processing pipeline."""
    processor = create_processor(processor_name)
    result = processor.process(items)
    return result
'''


@pytest.fixture
def sample_js_source() -> str:
    """Sample JavaScript source code for parser testing."""
    return '''import { useState, useEffect } from "react";
import axios from "axios";

const API_URL = "https://api.example.com";

export class UserService {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async getUser(id) {
        const response = await axios.get(`${this.baseUrl}/users/${id}`);
        return response.data;
    }

    async createUser(data) {
        const response = await axios.post(`${this.baseUrl}/users`, data);
        return response.data;
    }
}

export function formatName(first, last) {
    return `${first} ${last}`;
}

export const fetchData = async (url) => {
    const response = await fetch(url);
    return response.json();
};
'''
