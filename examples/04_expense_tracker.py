"""
Example Project 4: Expense Tracker (natural language -> structured data)

Domain: personal finance - chat naturally, get structured records.

What it shows:
  - LLM structured extraction: "spent 250 on coffee" -> {item, amount, category}
    via llm.with_structured_output (lesson 7's router trick, applied to data)
  - MongoDB as the system of record (optional - falls back to memory)
  - Querying your own data back in natural language (Python-side aggregation)

Run:  uv run examples/04_expense_tracker.py
Deps: OpenAI key only (Mongo optional for persistence).
"""

import os
import sys
from pathlib import Path

# make `modules` importable when run as: uv run examples/04_expense_tracker.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

load_dotenv()

ROOT = Path(__file__).parent.parent


class Expense(BaseModel):
    """One extracted expense record.

    Attributes:
        item: what was bought, e.g. "coffee".
        amount: how much was spent (same currency the user mentioned).
        category: one of food, travel, shopping, bills, fun, other.
    """

    item: str
    amount: float
    category: str


class ExpenseStore:
    """Storage for expenses with MongoDB persistence and an in-memory fallback.

    Attributes:
        mongo: MongoUtil bound to the expenses collection, or None offline.
        memory: fallback list of expense dicts when Mongo is unavailable.
    """

    def __init__(self) -> None:
        """Tries to connect to MongoDB; falls back to an in-memory list."""
        self.mongo = None
        self.memory = []
        try:
            from modules.mongo_utils import MongoUtil

            uri = os.getenv("MONGO_URI", "mongodb://localhost:27017").split("?")[0]
            uri += "?serverSelectionTimeoutMS=2000&connectTimeoutMS=2000"
            self.mongo = MongoUtil(database="examples", collection="expenses", connection_string=uri)
            self.mongo.count_all()  # cheap probe
            print("[mongo] connected - expenses will persist")
        except Exception:
            self.mongo = None  # probe failed - undo the handle, use memory
            print("[mongo] offline - expenses kept in memory for this run only")

    def add(self, expense: Expense, note: str) -> None:
        """Stores one expense record.

        Args:
            expense: the extracted expense.
            note: the original sentence the user wrote.
        """
        record = {
            "item": expense.item,
            "amount": expense.amount,
            "category": expense.category,
            "note": note,
        }
        if self.mongo is not None:
            self.mongo.create(record)
        else:
            self.memory.append(record)

    def all_expenses(self) -> list:
        """Returns every stored expense record.

        Returns:
            List of expense dicts from Mongo or from the memory fallback.
        """
        if self.mongo is not None:
            return list(self.mongo.get_all().values())
        return list(self.memory)

    def total_by_category(self) -> dict:
        """Aggregates spending per category.

        Returns:
            Dict mapping category name to total amount.
        """
        totals = {}
        for record in self.all_expenses():
            category = record.get("category", "other")
            totals[category] = totals.get(category, 0) + float(record.get("amount", 0))
        return totals


def main() -> None:
    """Runs the demo: extract 3 expenses, then report totals by category."""
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    extractor = llm.with_structured_output(Expense)
    store = ExpenseStore()

    sentences = [
        "spent 250 rupees on filter coffee this morning",
        "paid 1200 for pizza dinner with friends",
        "booked train tickets for 3000",
    ]

    for sentence in sentences:
        expense = extractor.invoke(
            "Extract the expense from this sentence. Categories: food, travel, "
            "shopping, bills, fun, other.\n\nSentence: " + sentence
        )
        store.add(expense, sentence)
        print("added:", expense.item, "|", expense.amount, "|", expense.category)

    print()
    print("=== spending by category ===")
    for category, total in store.total_by_category().items():
        print("  " + category + ": " + str(total))

    print()
    print("records stored:", len(store.all_expenses()))


if __name__ == "__main__":
    main()
