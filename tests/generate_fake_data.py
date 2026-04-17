"""Generate fake customer & product JSON for RAG testing.

Run:  uv run python tests/generate_fake_data.py
Output: tests/fake_customers.json, tests/fake_products.json
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from faker import Faker

fake = Faker(["en_US", "de_DE", "fr_FR"])
fake.seed_instance(42)

TESTS_DIR = Path(__file__).parent
NUM_CUSTOMERS = 5000
NUM_PRODUCTS = 5000

TIERS = ["free", "pro", "enterprise"]
TIER_WEIGHTS = [0.5, 0.35, 0.15]

CATEGORIES = [
    "Electronics",
    "Office Supplies",
    "Software",
    "Networking",
    "Storage",
    "Peripherals",
]

SUPPLIERS = [
    "Apple",
    "Samsung",
    "Logitech",
    "Dell",
    "HP",
    "Bosch",
    "Siemens",
    "Sony",
    "Microsoft",
    "Lenovo",
]


def generate_customers(n: int) -> list[dict]:
    customers = []
    for i in range(1, n + 1):
        company = fake.company()
        country = fake.country()
        tier = fake.random_element(elements=OrderedDict(zip(TIERS, TIER_WEIGHTS)))
        created = fake.date_between(start_date="-3y", end_date="today")
        name = fake.name()
        email = fake.company_email()

        content = (
            f"{name} works at {company} in {country}. "
            f"Account tier: {tier}. Customer since {created.isoformat()}. "
            f"Contact: {email}."
        )

        customers.append(
            {
                "id": f"cust-{i:04d}",
                "name": name,
                "email": email,
                "company": company,
                "country": country,
                "tier": tier,
                "created_at": created.isoformat(),
                "content": content,
                "language": fake.random_element(elements=["en", "de", "fr"]),
            }
        )
    return customers


def generate_products(n: int) -> list[dict]:
    products = []
    for i in range(1, n + 1):
        name = fake.catch_phrase()
        category = fake.random_element(elements=CATEGORIES)
        supplier = fake.random_element(elements=SUPPLIERS)
        price = round(
            fake.pyfloat(min_value=9.99, max_value=2999.99, right_digits=2), 2
        )
        sku = fake.bothify(text="???-####").upper()
        in_stock = fake.boolean(chance_of_getting_true=75)
        released = fake.date_between(start_date="-2y", end_date="today")
        description = fake.paragraph(nb_sentences=3)

        content = (
            f"{name} — {category} by {supplier}. {description} "
            f"Price: ${price:.2f}. SKU: {sku}. "
            f"{'In stock' if in_stock else 'Out of stock'}. "
            f"Released {released.isoformat()}."
        )

        products.append(
            {
                "id": f"prod-{i:04d}",
                "name": name,
                "sku": sku,
                "category": category,
                "supplier": supplier,
                "price": price,
                "in_stock": in_stock,
                "released_at": released.isoformat(),
                "description": description,
                "content": content,
                "language": fake.random_element(elements=["en", "de", "fr"]),
            }
        )
    return products


# ── Cross-field confusion test data ─────────────────────────────────────────
# Deliberately crafted: attributes that DON'T combine the way a naive query
# would imply. E.g., "product 1001 from Samsung" — but 1001 is from Apple.


def generate_confusion_products() -> list[dict]:
    """Hand-crafted products designed to test attribute-binding precision."""
    return [
        {
            "id": "conf-1001",
            "name": "UltraWidget Pro",
            "sku": "UWP-1001",
            "category": "Electronics",
            "supplier": "Apple",
            "price": 999.00,
            "in_stock": True,
            "released_at": "2025-06-15",
            "description": "Premium widget with advanced features.",
            "content": "UltraWidget Pro — Electronics by Apple. Premium widget with advanced features. Price: $999.00. SKU: UWP-1001. In stock. Released 2025-06-15.",
            "language": "en",
        },
        {
            "id": "conf-1002",
            "name": "UltraWidget Lite",
            "sku": "UWL-1002",
            "category": "Electronics",
            "supplier": "Samsung",
            "price": 499.00,
            "in_stock": True,
            "released_at": "2025-08-01",
            "description": "Lightweight widget for everyday use.",
            "content": "UltraWidget Lite — Electronics by Samsung. Lightweight widget for everyday use. Price: $499.00. SKU: UWL-1002. In stock. Released 2025-08-01.",
            "language": "en",
        },
        {
            "id": "conf-1003",
            "name": "PowerHub Max",
            "sku": "PHM-1003",
            "category": "Networking",
            "supplier": "Dell",
            "price": 349.00,
            "in_stock": False,
            "released_at": "2024-11-20",
            "description": "Enterprise networking hub with 48 ports.",
            "content": "PowerHub Max — Networking by Dell. Enterprise networking hub with 48 ports. Price: $349.00. SKU: PHM-1003. Out of stock. Released 2024-11-20.",
            "language": "en",
        },
        {
            "id": "conf-1004",
            "name": "PowerHub Mini",
            "sku": "PHM-1004",
            "category": "Networking",
            "supplier": "Logitech",
            "price": 89.00,
            "in_stock": True,
            "released_at": "2025-03-10",
            "description": "Compact networking hub for home office.",
            "content": "PowerHub Mini — Networking by Logitech. Compact networking hub for home office. Price: $89.00. SKU: PHM-1004. In stock. Released 2025-03-10.",
            "language": "en",
        },
        {
            "id": "conf-1005",
            "name": "DataVault 500",
            "sku": "DV-1005",
            "category": "Storage",
            "supplier": "Samsung",
            "price": 159.00,
            "in_stock": True,
            "released_at": "2025-01-05",
            "description": "500GB portable SSD with encryption.",
            "content": "DataVault 500 — Storage by Samsung. 500GB portable SSD with encryption. Price: $159.00. SKU: DV-1005. In stock. Released 2025-01-05.",
            "language": "en",
        },
        {
            "id": "conf-1006",
            "name": "DataVault 2000",
            "sku": "DV-1006",
            "category": "Storage",
            "supplier": "Apple",
            "price": 399.00,
            "in_stock": False,
            "released_at": "2025-04-22",
            "description": "2TB desktop SSD for creative professionals.",
            "content": "DataVault 2000 — Storage by Apple. 2TB desktop SSD for creative professionals. Price: $399.00. SKU: DV-1006. Out of stock. Released 2025-04-22.",
            "language": "en",
        },
    ]


def main() -> None:
    customers = generate_customers(NUM_CUSTOMERS)
    products = generate_products(NUM_PRODUCTS)
    confusion = generate_confusion_products()

    cust_path = TESTS_DIR / "fake_customers.json"
    prod_path = TESTS_DIR / "fake_products.json"
    conf_path = TESTS_DIR / "fake_confusion_products.json"

    cust_path.write_text(json.dumps(customers, indent=2, ensure_ascii=False))
    prod_path.write_text(json.dumps(products, indent=2, ensure_ascii=False))
    conf_path.write_text(json.dumps(confusion, indent=2, ensure_ascii=False))

    print(f"Wrote {len(customers)} customers  → {cust_path}")
    print(f"Wrote {len(products)} products   → {prod_path}")
    print(f"Wrote {len(confusion)} confusion  → {conf_path}")


if __name__ == "__main__":
    main()
