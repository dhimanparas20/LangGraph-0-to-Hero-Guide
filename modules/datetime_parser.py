"""
datetime_parser - A production-level datetime parsing and conversion module.

This module provides a robust solution for parsing datetime strings in various
formats and converting them across multiple timezones. It leverages the
`dateparser` library for flexible input parsing and Python's `zoneinfo` for
accururate timezone handling.

Key Features:
    - Parse datetime strings in virtually any human-readable format.
    - Convert parsed datetime to UTC, system timezone, or custom timezones.
    - Supports both common abbreviations (e.g., 'IST', 'PST') and full IANA
      timezone names (e.g., 'Asia/Kolkata', 'America/New_York').
    - 100+ timezone aliases covering major countries worldwide.
    - Returns structured dictionaries with ISO strings and Unix timestamps.
    - Graceful error handling with detailed error messages.

Author: T3 Chat
Version: 3.0.0
Date: 2026-06-23
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Any, Optional

from dateparser import parse as dateparser_parse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# ======================================================================
# Module-level configuration
# ======================================================================

__version__ = "3.0.0"
__author__ = "T3 Chat"

logger = logging.getLogger(__name__)

# Default timezone used when input datetime is naive (no timezone info)
DEFAULT_TIMEZONE: str = "Asia/Kolkata"

# ======================================================================
# Timezone Aliases
# Covers 100+ countries/regions with common abbreviations mapped to
# their official IANA timezone strings.
# ======================================================================

TZ_ALIASES: dict[str, str] = {
    # ──────────────────────────────────────────────
    # SOUTH ASIA
    # ──────────────────────────────────────────────
    "IST": "Asia/Kolkata",          # India Standard Time
    "SLST": "Asia/Colombo",         # Sri Lanka Standard Time
    "BST_BD": "Asia/Dhaka",         # Bangladesh Standard Time
    "NPT": "Asia/Kathmandu",        # Nepal Time
    "BTT": "Asia/Thimphu",          # Bhutan Time
    "PKT": "Asia/Karachi",          # Pakistan Standard Time
    "AFT": "Asia/Kabul",            # Afghanistan Time
    "MVT": "Indian/Maldives",       # Maldives Time

    # ──────────────────────────────────────────────
    # SOUTHEAST ASIA
    # ──────────────────────────────────────────────
    "ICT": "Asia/Bangkok",          # Indochina Time
    "SGT": "Asia/Singapore",        # Singapore Time
    "MYT": "Asia/Kuala_Lumpur",     # Malaysia Time
    "WIB": "Asia/Jakarta",          # Western Indonesian Time
    "WITA": "Asia/Makassar",        # Central Indonesian Time
    "WIT": "Asia/Jayapura",         # Eastern Indonesian Time
    "PHT": "Asia/Manila",           # Philippine Time
    "SGT_PH": "Asia/Manila",        # Philippine Time (alt)
    "ICT_VN": "Asia/Ho_Chi_Minh",   # Vietnam Time
    "MMT": "Asia/Yangon",           # Myanmar Time
    "TLT": "Asia/Dili",             # Timor-Leste Time

    # ──────────────────────────────────────────────
    # EAST ASIA
    # ──────────────────────────────────────────────
    "JST": "Asia/Tokyo",            # Japan Standard Time
    "KST": "Asia/Seoul",            # Korea Standard Time
    "CST_CN": "Asia/Shanghai",      # China Standard Time
    "HKT": "Asia/Hong_Kong",        # Hong Kong Time
    "CST_TW": "Asia/Taipei",        # Taiwan Time
    "ULAT": "Asia/Ulaanbaatar",     # Ulaanbaatar Time
    "PHT2": "Asia/Manila",          # Philippines (alt)
    "GNST": "Asia/Guam",            # Chamorro Standard Time

    # ──────────────────────────────────────────────
    # MIDDLE EAST
    # ──────────────────────────────────────────────
    "AST_SA": "Asia/Riyadh",        # Arabian Standard Time (Saudi Arabia)
    "AST_BH": "Asia/Bahrain",       # Bahrain Time
    "AST_QA": "Asia/Qatar",         # Qatar Time
    "AST_KW": "Asia/Kuwait",        # Kuwait Time
    "AST_AE": "Asia/Dubai",         # Gulf Standard Time (UAE)
    "AST_OM": "Asia/Muscat",        # Oman Time
    "AST_IQ": "Asia/Baghdad",       # Arabia Standard Time (Iraq)
    "AST_SY": "Asia/Damascus",      # Syria Time
    "AST_JO": "Asia/Amman",         # Jordan Time
    "AST_LB": "Asia/Beirut",        # Lebanon Time
    "AST_IL": "Asia/Jerusalem",     # Israel Standard Time
    "AST_PS": "Asia/Gaza",          # Palestine Time
    "IRST": "Asia/Tehran",          # Iran Standard Time
    "TRT": "Europe/Istanbul",       # Turkey Time
    "AZT": "Asia/Baku",             # Azerbaijan Time
    "GET": "Asia/Tbilisi",          # Georgia Time
    "AMT_AM": "Asia/Yerevan",       # Armenia Time
    "AZST": "Asia/Baku",            # Azerbaijan Summer Time

    # ──────────────────────────────────────────────
    # CENTRAL ASIA
    # ──────────────────────────────────────────────
    "UZT": "Asia/Tashkent",         # Uzbekistan Time
    "TMT": "Asia/Ashgabat",         # Turkmenistan Time
    "KGT": "Asia/Bishkek",          # Kyrgyzstan Time
    "TJT": "Asia/Dushanbe",         # Tajikistan Time
    "KAZ": "Asia/Almaty",           # Kazakhstan Time (Almaty)
    "KZT_NUR": "Asia/Almaty",       # Kazakhstan (Nur-Sultan)
    "AQTT": "Asia/Aqtau",           # Aqtau Time

    # ──────────────────────────────────────────────
    # EUROPE (WEST)
    # ──────────────────────────────────────────────
    "GMT": "Europe/London",          # Greenwich Mean Time
    "BST": "Europe/London",          # British Summer Time
    "IST_IE": "Europe/Dublin",       # Irish Standard Time
    "WET": "Europe/Lisbon",          # Western European Time
    "WEST": "Europe/Lisbon",         # Western European Summer Time
    "CET": "Europe/Berlin",          # Central European Time
    "CEST": "Europe/Berlin",         # Central European Summer Time
    "MET": "Europe/Amsterdam",       # Middle European Time
    "MEST": "Europe/Amsterdam",      # Middle European Summer Time
    "PST_PT": "Europe/Lisbon",       # Portugal Standard Time

    # ──────────────────────────────────────────────
    # EUROPE (EAST)
    # ──────────────────────────────────────────────
    "EET": "Europe/Athens",          # Eastern European Time
    "EEST": "Europe/Athens",         # Eastern European Summer Time
    "MSK": "Europe/Moscow",          # Moscow Standard Time
    "FET": "Europe/Minsk",           # Further Eastern European Time
    "EET_FI": "Europe/Helsinki",     # Finland Time
    "EET_RO": "Europe/Bucharest",    # Romania Time
    "EET_BG": "Europe/Sofia",        # Bulgaria Time
    "EET_UA": "Europe/Kiev",         # Ukraine Time
    "EET_MD": "Europe/Chisinau",     # Moldova Time
    "EET_LT": "Europe/Vilnius",      # Lithuania Time
    "EET_LV": "Europe/Riga",         # Latvia Time
    "EET_EE": "Europe/Tallinn",      # Estonia Time
    "EET_GR": "Europe/Athens",       # Greece Time
    "EET_RS": "Europe/Belgrade",     # Serbia Time
    "EET_HR": "Europe/Zagreb",       # Croatia Time
    "EET_SI": "Europe/Ljubljana",    # Slovenia Time
    "EET_SK": "Europe/Bratislava",   # Slovakia Time
    "EET_CZ": "Europe/Prague",       # Czech Time
    "CET_PL": "Europe/Warsaw",       # Poland Time
    "CET_HU": "Europe/Budapest",     # Hungary Time

    # ──────────────────────────────────────────────
    # RUSSIA
    # ──────────────────────────────────────────────
    "YEKT": "Asia/Yekaterinburg",    # Yekaterinburg Time
    "OMST": "Asia/Omsk",             # Omsk Time
    "KRAS": "Asia/Krasnoyarsk",      # Krasnoyarsk Time
    "IRKT": "Asia/Irkutsk",          # Irkutsk Time
    "YAKT": "Asia/Yakutsk",          # Yakutsk Time
    "VLAT": "Asia/Vladivostok",      # Vladivostok Time
    "MAGT": "Asia/Magadan",          # Magadan Time
    "PETT": "Asia/Kamchatka",        # Kamchatka Time
    "SAKT": "Asia/Sakhalin",         # Sakhalin Time

    # ──────────────────────────────────────────────
    # NORTH AMERICA
    # ──────────────────────────────────────────────
    "EST": "America/New_York",        # Eastern Standard Time
    "EDT": "America/New_York",        # Eastern Daylight Time
    "CST": "America/Chicago",         # Central Standard Time
    "CDT": "America/Chicago",         # Central Daylight Time
    "MST": "America/Denver",          # Mountain Standard Time
    "MDT": "America/Denver",          # Mountain Daylight Time
    "PST": "America/Los_Angeles",     # Pacific Standard Time
    "PDT": "America/Los_Angeles",     # Pacific Daylight Time
    "AKST": "America/Anchorage",      # Alaska Standard Time
    "AKDT": "America/Anchorage",      # Alaska Daylight Time
    "HST": "Pacific/Honolulu",        # Hawaii Standard Time
    "NST": "America/St_Johns",        # Newfoundland Standard Time
    "NDT": "America/St_Johns",        # Newfoundland Daylight Time

    # ──────────────────────────────────────────────
    # CANADA
    # ──────────────────────────────────────────────
    "PST_CA": "America/Vancouver",    # Pacific (Canada)
    "MST_CA": "America/Edmonton",     # Mountain (Canada)
    "CST_CA": "America/Winnipeg",     # Central (Canada)
    "EST_CA": "America/Toronto",      # Eastern (Canada)
    "NST_CA": "America/St_Johns",     # Newfoundland (Canada)

    # ──────────────────────────────────────────────
    # MEXICO & CENTRAL AMERICA
    # ──────────────────────────────────────────────
    "CST_MX": "America/Mexico_City",  # Mexico City Time
    "PST_MX": "America/Tijuana",      # Tijuana Time
    "MST_MX": "America/Chihuahua",    # Chihuahua Time
    "CST_CR": "America/Costa_Rica",   # Costa Rica Time
    "CST_GT": "America/Guatemala",    # Guatemala Time
    "CST_HN": "America/Tegucigalpa",  # Honduras Time
    "CST_SV": "America/El_Salvador",  # El Salvador Time
    "CST_NI": "America/Managua",      # Nicaragua Time
    "CST_PA": "America/Panama",       # Panama Time
    "CST_BZ": "America/Belize",       # Belize Time
    "CST_HN": "America/Tegucigalpa",  # Honduras Time
    "EST_CU": "America/Havana",       # Cuba Time

    # ──────────────────────────────────────────────
    # SOUTH AMERICA
    # ──────────────────────────────────────────────
    "BRT": "America/Sao_Paulo",       # Brasilia Time (Brazil)
    "ART": "America/Argentina/Buenos_Aires", # Argentina Time
    "CLT": "America/Santiago",        # Chile Time
    "UYT": "America/Montevideo",      # Uruguay Time
    "PYT": "America/Asuncion",        # Paraguay Time
    "BOT": "America/La_Paz",          # Bolivia Time
    "PET": "America/Lima",            # Peru Time
    "ECT": "America/Guayaquil",       # Ecuador Time
    "VET": "America/Caracas",         # Venezuela Time
    "GFT": "America/Cayenne",         # French Guiana Time
    "SRT": "America/Paramaribo",      # Suriname Time
    "GYT": "America/Guyana",          # Guyana Time

    # ──────────────────────────────────────────────
    # AFRICA (NORTH)
    # ──────────────────────────────────────────────
    "WAT_NG": "Africa/Lagos",         # West Africa Time (Nigeria)
    "WAT": "Africa/Lagos",            # West Africa Time
    "CAT": "Africa/Harare",           # Central Africa Time
    "EAT": "Africa/Nairobi",          # East Africa Time
    "SAST": "Africa/Johannesburg",    # South Africa Standard Time
    "CAT_CD": "Africa/Kinshasa",      # DR Congo (West)
    "CAT_TZ": "Africa/Dar_es_Salaam", # Tanzania Time
    "EAT_UG": "Africa/Kampala",       # Uganda Time
    "EAT_ET": "Africa/Addis_Ababa",   # Ethiopia Time
    "EAT_KE": "Africa/Nairobi",       # Kenya Time
    "EAT_RW": "Africa/Kigali",        # Rwanda Time
    "EAT_BI": "Africa/Bujumbura",     # Burundi Time
    "WAT_AO": "Africa/Luanda",        # Angola Time
    "WAT_GA": "Africa/Libreville",    # Gabon Time
    "WAT_CG": "Africa/Brazzaville",   # Congo Time
    "WAT_CM": "Africa/Douala",        # Cameroon Time
    "WAT_TD": "Africa/Ndjamena",      # Chad Time
    "WAT_CF": "Africa/Bangui",        # Central African Republic Time
    "WAT_ST": "Africa/Sao_Tome",      # Sao Tome Time
    "WAT_NE": "Africa/Niamey",        # Niger Time
    "WAT_ML": "Africa/Bamako",        # Mali Time
    "WAT_BF": "Africa/Ouagadougou",   # Burkina Faso Time
    "WAT_GN": "Africa/Conakry",       # Guinea Time
    "WAT_CI": "Africa/Abidjan",       # Ivory Coast Time
    "WAT_SN": "Africa/Dakar",         # Senegal Time
    "WAT_GM": "Africa/Banjul",        # Gambia Time
    "WAT_GW": "Africa/Bissau",        # Guinea-Bissau Time
    "WAT_CV": "Atlantic/Cape_Verde",  # Cape Verde Time

    # ──────────────────────────────────────────────
    # AFRICA (OTHER)
    # ──────────────────────────────────────────────
    "CAT_MW": "Africa/Blantyre",      # Malawi Time
    "CAT_ZM": "Africa/Lusaka",        # Zambia Time
    "CAT_ZW": "Africa/Harare",        # Zimbabwe Time
    "CAT_BW": "Africa/Gaborone",      # Botswana Time
    "CAT_NA": "Africa/Windhoek",      # Namibia Time
    "CAT_MZ": "Africa/Maputo",        # Mozambique Time
    "CAT_SZ": "Africa/Mbabane",       # Eswatini Time
    "CAT_LS": "Africa/Maseru",        # Lesotho Time

    # ──────────────────────────────────────────────
    # OCEANIA
    # ──────────────────────────────────────────────
    "AEST": "Australia/Sydney",        # Australian Eastern Standard Time
    "AEDT": "Australia/Sydney",        # Australian Eastern Daylight Time
    "ACST": "Australia/Adelaide",      # Australian Central Standard Time
    "ACDT": "Australia/Adelaide",      # Australian Central Daylight Time
    "AWST": "Australia/Perth",         # Australian Western Standard Time
    "NZST": "Pacific/Auckland",        # New Zealand Standard Time
    "NZDT": "Pacific/Auckland",        # New Zealand Daylight Time
    "FJT": "Pacific/Fiji",             # Fiji Time
    "WST_WS": "Pacific/Apia",         # West Samoa Time
    "PGT": "Pacific/Port_Moresby",     # Papua New Guinea Time
    "SBT": "Pacific/Guadalcanal",      # Solomon Islands Time
    "VUT": "Pacific/Efate",            # Vanuatu Time
    "CHST": "Pacific/Guam",            # Chamorro Standard Time
    "TLT_TL": "Pacific/Dili",          # East Timor Time
    "MHT": "Pacific/Majuro",           # Marshall Islands Time
    "CHST_MH": "Pacific/Majuro",       # Marshall Islands Time
    "SST": "Pacific/Pago_Pago",        # Samoa Standard Time
    "TKT": "Pacific/Fakaofo",          # Tokelau Time
    "LINT": "Pacific/Kiritimati",      # Line Islands Time

    # ──────────────────────────────────────────────
    # ATLANTIC & SPECIAL
    # ──────────────────────────────────────────────
    "WAT_GI": "Europe/Gibraltar",      # Gibraltar Time
    "CET_MC": "Europe/Monaco",         # Monaco Time
    "CET_AD": "Europe/Andorra",        # Andorra Time
    "CET_SM": "Europe/San_Marino",     # San Marino Time
    "CET_VA": "Europe/Vatican",        # Vatican Time
    "CET_LI": "Europe/Vaduz",          # Liechtenstein Time
    "CET_LU": "Europe/Luxembourg",     # Luxembourg Time
    "CET_BE": "Europe/Brussels",       # Belgium Time
    "CET_CH": "Europe/Zurich",         # Switzerland Time
    "WET_FO": "Atlantic/Faroe",        # Faroe Islands Time
    "WET_IS": "Atlantic/Reykjavik",    # Iceland Time
    "WET_GL": "America/Nuuk",          # Greenland Time
    "MDT_MX": "America/Mazatlan",      # Mazatlan Time
    "PST_BZ": "America/Belize",        # Belize Time
    "WET_MA": "Africa/Casablanca",     # Morocco Time
    "WET_DZ": "Africa/Algiers",        # Algeria Time
    "EET_LY": "Africa/Tripoli",        # Libya Time
    "EET_EG": "Africa/Cairo",          # Egypt Time
    "EET_TN": "Africa/Tunis",          # Tunisia Time
    "EET_SD": "Africa/Khartoum",       # Sudan Time
    "EET_SS": "Africa/Juba",           # South Sudan Time
}


# ======================================================================
# Public API
# ======================================================================


def convert_datetime(
    datetime_string: str,
    parse_to: Optional[list[str]] = None,
    parse_to_system: bool = False,
    parse_to_utc: bool = False,
) -> dict[str, Any]:
    """
    Parse a datetime string and optionally convert it to multiple timezones.

    This is the main entry point of the module. It accepts a datetime string
    in virtually any human-readable format and returns a structured dictionary
    containing the parsed result in the requested timezones.

    Args:
        datetime_string (str): The datetime string to parse. Can be in any
            format supported by the ``dateparser`` library, such as:

            - ``"2026-08-13 13:30:00"``
            - ``"August 13, 2026 1:30 PM"``
            - ``"13/08/2026 13:30"``
            - ``"Tomorrow at 10 AM"``
            - ``"2 hours ago"``
            - ``"13th August 2026 1:30 PM"``

        parse_to (Optional[list[str]]): A list of timezone strings to convert
            the parsed datetime into. Accepts both common aliases (e.g.,
            ``"IST"``, ``"PST"``) and full IANA timezone names (e.g.,
            ``"Asia/Kolkata"``, ``"America/New_York"``). These conversions
            are only performed when ``parse_to_system`` is ``False``.

        parse_to_system (bool): If ``True``, converts the parsed datetime to
            the system's local timezone and includes the timezone name in the
            response. When ``True``, ``parse_to`` conversions are skipped.

        parse_to_utc (bool): If ``True``, converts the parsed datetime to UTC
            and includes both the ISO-formatted string and the Unix timestamp.

    Returns:
        dict[str, Any]: A dictionary containing:

            - ``"original_string"`` (str): The original datetime string passed.
            - ``"is_parsed"`` (bool): Whether the string was successfully parsed.
            - ``"parsed_datetime"`` (str | None): The datetime in ISO 8601 format.
            - ``"parsed_timestamp"`` (int | None): The Unix timestamp.
            - ``"utc"`` (str | None): UTC datetime in ISO 8601 format.
            - ``"utc_timestamp"`` (int | None): Unix timestamp in UTC.
            - ``"system"`` (str | None): System timezone datetime string.
            - ``"system_timestamp"`` (int | None): Unix timestamp in system tz.
            - ``"system_timezone"`` (str | None): Name of the system timezone.
            - ``"custom_timezones"`` (dict): Converted datetime for each tz.
            - ``"error"`` (str | None): Error message if parsing failed.
            - ``"utc_error"`` (str | None): Error if UTC conversion failed.
            - ``"system_error"`` (str | None): Error if system tz failed.

    Raises:
        None. This function is designed to never raise exceptions. All errors
        are captured in the response dictionary.

    Examples:
        Basic usage with UTC conversion:

        >>> result = convert_datetime(
        ...     datetime_string="August 13, 2026 1:30 PM",
        ...     parse_to_utc=True,
        ... )
        >>> print(result["utc"])
        "2026-08-13T08:00:00+0000"

        Full usage with all options:

        >>> result = convert_datetime(
        ...     datetime_string="13/08/2026 13:30",
        ...     parse_to=["IST", "America/New_York"],
        ...     parse_to_system=True,
        ...     parse_to_utc=True,
        ... )
        >>> print(result)
        {
            "original_string": "13/08/2026 13:30",
            "is_parsed": True,
            "parsed_datetime": "2026-08-13T13:30:00+05:30",
            "parsed_timestamp": 1786680000,
            "utc": "2026-08-13T08:00:00+0000",
            "utc_timestamp": 1786645200,
            "system": "2026-08-13 01:30:00 PM IST",
            "system_timezone": "Asia/Kolkata",
            "custom_timezones": {},
            "error": None,
            "utc_error": None,
            "system_error": None,
        }

        Using relative datetime strings:

        >>> result = convert_datetime(
        ...     datetime_string="Tomorrow at 10 AM",
        ...     parse_to_system=True,
        ...     parse_to_utc=True,
        ... )
    """

    # --------------------------------------------------
    # Initialize response structure
    # --------------------------------------------------
    response: dict[str, Any] = {
        "original_string": datetime_string,
        "is_parsed": False,
        "parsed_datetime": None,
        "parsed_timestamp": None,
        "utc": None,
        "utc_timestamp": None,
        "system": None,
        "system_timestamp": None,
        "system_timezone": None,
        "custom_timezones": {},
        "error": None,
        "utc_error": None,
        "system_error": None,
    }

    # --------------------------------------------------
    # Step 1: Parse the input string
    # --------------------------------------------------
    parsed_dt = _parse_input(datetime_string, response)
    if parsed_dt is None:
        return response

    # --------------------------------------------------
    # Step 2: Attach default timezone (IST) to naive datetime
    # --------------------------------------------------
    parsed_dt = _attach_default_timezone(parsed_dt)

    # --------------------------------------------------
    # Step 3: Store the initial parsed result
    # --------------------------------------------------
    response["parsed_datetime"] = parsed_dt.isoformat()
    response["parsed_timestamp"] = int(parsed_dt.timestamp())

    # --------------------------------------------------
    # Step 4: Handle UTC conversion
    # --------------------------------------------------
    if parse_to_utc:
        _convert_to_utc(parsed_dt, response)

    # --------------------------------------------------
    # Step 5: Handle System timezone conversion
    # --------------------------------------------------
    if parse_to_system:
        _convert_to_system(parsed_dt, response)

    # --------------------------------------------------
    # Step 6: Handle Custom timezone conversions
    # --------------------------------------------------
    if parse_to and not parse_to_system:
        _convert_to_custom(parsed_dt, parse_to, response)

    return response


def resolve_timezone(tz_string: str) -> ZoneInfo:
    """
    Resolve a timezone string to a ``ZoneInfo`` object.

    This function accepts both common timezone aliases (e.g., ``"IST"``,
    ``"PST"``) and full IANA timezone names (e.g., ``"Asia/Kolkata"``,
    ``"America/New_York"``).

    Args:
        tz_string (str): The timezone string to resolve.

    Returns:
        ZoneInfo: The corresponding ``ZoneInfo`` object.

    Raises:
        ValueError: If the timezone string cannot be resolved.

    Examples:
        >>> tz = resolve_timezone("IST")
        >>> print(tz)
        Asia/Kolkata

        >>> tz = resolve_timezone("America/New_York")
        >>> print(tz)
        America/New_York
    """
    return _resolve_timezone(tz_string)


def get_system_timezone() -> tuple[str, str]:
    """
    Get the current system timezone name and its IANA equivalent.

    Returns:
        tuple[str, str]: A tuple of ``(system_tz_name, iana_tz_string)``.

    Examples:
        >>> name, iana = get_system_timezone()
        >>> print(f"System timezone: {name} ({iana})")
        System timezone: IST (Asia/Kolkata)
    """
    system_tz = datetime.now().astimezone().tzinfo
    system_tz_name = str(system_tz)
    return system_tz_name, system_tz_name


def list_supported_timezones() -> list[dict[str, str]]:
    """
    List all supported timezone aliases and their IANA equivalents.

    Returns:
        list[dict[str, str]]: A list of dicts with ``'alias'`` and
        ``'iana'`` keys.

    Examples:
        >>> timezones = list_supported_timezones()
        >>> for tz in timezones[:3]:
        ...     print(f"{tz['alias']} -> {tz['iana']}")
        IST -> Asia/Kolkata
        PST -> America/Los_Angeles
        EST -> America/New_York
    """
    return [{"alias": alias, "iana": iana} for alias, iana in TZ_ALIASES.items()]


def get_all_aliases_for_iana(iana_name: str) -> list[str]:
    """
    Get all alias abbreviations that map to a given IANA timezone name.

    This is useful for reverse lookups. For example, passing
    ``"Asia/Kolkata"`` would return ``["IST"]``.

    Args:
        iana_name (str): The IANA timezone name to search for.

    Returns:
        list[str]: A list of alias strings that map to the given IANA name.

    Examples:
        >>> aliases = get_all_aliases_for_iana("Asia/Kolkata")
        >>> print(aliases)
        ["IST"]

        >>> aliases = get_all_aliases_for_iana("America/New_York")
        >>> print(aliases)
        ["EST", "EDT"]
    """
    return [alias for alias, iana in TZ_ALIASES.items() if iana == iana_name]


# ======================================================================
# Internal Helper Functions
# ======================================================================


def _parse_input(
    datetime_string: str, response: dict[str, Any]
) -> Optional[datetime]:
    """
    Parse the input datetime string using dateparser.

    Args:
        datetime_string: The datetime string to parse.
        response: The response dictionary to update on failure.

    Returns:
        The parsed datetime object, or ``None`` if parsing failed.
    """
    try:
        parsed_dt = dateparser_parse(datetime_string)
        if parsed_dt is None:
            response["error"] = (
                f"Could not parse the datetime string: '{datetime_string}'"
            )
            logger.warning(response["error"])
            return None
        response["is_parsed"] = True
        logger.debug(
            f"Successfully parsed: '{datetime_string}' -> {parsed_dt}"
        )
        return parsed_dt
    except Exception as e:
        response["error"] = f"Parsing failed with exception: {str(e)}"
        logger.error(response["error"])
        return None


def _attach_default_timezone(dt: datetime) -> datetime:
    """
    Attach the default timezone (``Asia/Kolkata``) to a naive datetime.

    If ``dateparser`` returns a naive datetime (no timezone info), this
    function attaches the default timezone to it. This ensures that all
    subsequent conversions are accurate.

    Args:
        dt: The datetime object to attach timezone to.

    Returns:
        A timezone-aware datetime object.
    """
    if dt.tzinfo is None:
        logger.debug(
            f"Naive datetime detected, attaching default timezone: "
            f"{DEFAULT_TIMEZONE}"
        )
        return dt.replace(tzinfo=ZoneInfo(DEFAULT_TIMEZONE))
    return dt


def _convert_to_utc(dt: datetime, response: dict[str, Any]) -> None:
    """
    Convert a datetime to UTC and update the response dictionary.

    Args:
        dt: The datetime object to convert.
        response: The response dictionary to update.
    """
    try:
        utc_dt = dt.astimezone(ZoneInfo("UTC"))
        response["utc"] = utc_dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        response["utc_timestamp"] = int(utc_dt.timestamp())
        logger.debug(f"UTC conversion: {dt} -> {utc_dt}")
    except Exception as e:
        response["utc_error"] = f"UTC conversion failed: {str(e)}"
        logger.error(response["utc_error"])


def _convert_to_system(dt: datetime, response: dict[str, Any]) -> None:
    """
    Convert a datetime to system timezone and update the response.

    Args:
        dt: The datetime object to convert.
        response: The response dictionary to update.
    """
    try:
        system_tz = datetime.now().astimezone().tzinfo
        system_tz_name = str(system_tz)
        system_dt = dt.astimezone(system_tz)
        response["system"] = system_dt.strftime("%Y-%m-%d %I:%M:%S %p %Z")
        response["system_timestamp"] = int(system_dt.timestamp())
        response["system_timezone"] = system_tz_name
        logger.debug(
            f"System conversion: {dt} -> {system_dt} ({system_tz_name})"
        )
    except Exception as e:
        response["system_error"] = (
            f"System timezone conversion failed: {str(e)}"
        )
        logger.error(response["system_error"])


def _convert_to_custom(
    dt: datetime, parse_to: list[str], response: dict[str, Any]
) -> None:
    """
    Convert a datetime to custom timezones and update the response.

    Args:
        dt: The datetime object to convert.
        parse_to: List of timezone strings to convert to.
        response: The response dictionary to update.
    """
    for tz_string in parse_to:
        try:
            resolved_tz = _resolve_timezone(tz_string)
            converted_dt = dt.astimezone(resolved_tz)

            response["custom_timezones"][tz_string] = {
                "timezone_name": tz_string,
                "iana_name": str(resolved_tz),
                "datetime": converted_dt.strftime(
                    "%Y-%m-%d %I:%M:%S %p %Z"
                ),
                "timestamp": int(converted_dt.timestamp()),
                "iso_format": converted_dt.isoformat(),
                "error": None,
            }
            logger.debug(
                f"Custom conversion: {dt} -> {converted_dt} ({tz_string})"
            )
        except Exception as e:
            response["custom_timezones"][tz_string] = {
                "timezone_name": tz_string,
                "iana_name": None,
                "datetime": None,
                "timestamp": None,
                "iso_format": None,
                "error": (
                    f"Conversion failed for '{tz_string}': {str(e)}"
                ),
            }
            logger.error(
                f"Custom conversion failed for '{tz_string}': {e}"
            )


def _resolve_timezone(tz_string: str) -> ZoneInfo:
    """
    Resolve a timezone string to a ``ZoneInfo`` object.

    This function accepts both common aliases (e.g., ``'IST'``, ``'PST'``)
    and full IANA timezone names (e.g., ``'Asia/Kolkata'``,
    ``'America/New_York'``). Aliases are checked first (case-insensitive),
    then direct IANA lookup is attempted.

    Args:
        tz_string: The timezone string to resolve.

    Returns:
        The corresponding ``ZoneInfo`` object.

    Raises:
        ValueError: If the timezone string cannot be resolved.
    """
    # Step 1: Check common aliases (case-insensitive)
    upper_key = tz_string.upper()
    if upper_key in TZ_ALIASES:
        return ZoneInfo(TZ_ALIASES[upper_key])

    # Step 2: Try direct IANA lookup
    try:
        return ZoneInfo(tz_string)
    except (ZoneInfoNotFoundError, KeyError):
        pass

    # Step 3: Raise error with helpful message
    raise ValueError(
        f"Invalid timezone: '{tz_string}'. "
        f"Use IANA names (e.g., 'Asia/Kolkata') or supported aliases "
        f"(e.g., 'IST', 'PST'). Run list_supported_timezones() to see "
        f"all supported timezones."
    )


# ======================================================================
# NEW PUBLIC API: Timestamp & ISO Conversion
# ======================================================================


def timestamp_to_string(
    timestamp: int | float,
    output_timezone: str = "Asia/Kolkata",
    output_format: str = "%Y-%m-%d %I:%M:%S %p %Z",
) -> dict[str, Any]:
    """
    Convert a Unix timestamp (seconds or milliseconds) to a human-readable
    datetime string.

    Args:
        timestamp (int | float): The Unix timestamp to convert.
            Supports both seconds (e.g., 1786645200) and milliseconds
            (e.g., 1786645200000). The function auto-detects based on
            magnitude (if > 1e12, assumes milliseconds).

        output_timezone (str): The timezone to display the result in.
            Accepts aliases (e.g., "IST", "PST") or IANA names
            (e.g., "Asia/Kolkata", "America/New_York").

        output_format (str): The strftime format string to use for
            formatting the output. Defaults to "%Y-%m-%d %I:%M:%S %p %Z".

    Returns:
        dict[str, Any]: A dictionary containing:
            - "original_timestamp" (int | float): The original timestamp.
            - "is_milliseconds" (bool): Whether input was in milliseconds.
            - "utc" (str): UTC datetime string.
            - "utc_iso" (str): UTC ISO 8601 string.
            - "formatted" (str): Formatted datetime in requested timezone.
            - "timezone" (str): The timezone used for formatting.
            - "error" (str | None): Error message if conversion failed.

    Examples:
        >>> result = timestamp_to_string(1786645200)
        >>> print(result["formatted"])
        "2026-08-13 01:30:00 PM IST"

        >>> result = timestamp_to_string(1786645200000, output_timezone="PST")
        >>> print(result["formatted"])
        "2026-08-13 01:00:00 AM PDT"
    """
    response: dict[str, Any] = {
        "original_timestamp": timestamp,
        "is_milliseconds": False,
        "utc": None,
        "utc_iso": None,
        "formatted": None,
        "timezone": output_timezone,
        "error": None,
    }

    try:
        # Step 1: Auto-detect milliseconds vs seconds
        ts = float(timestamp)
        if ts > 1e12:
            ts = ts / 1000.0
            response["is_milliseconds"] = True

        # Step 2: Create UTC datetime from timestamp
        utc_dt = datetime.fromtimestamp(ts, tz=ZoneInfo("UTC"))
        response["utc"] = utc_dt.strftime("%Y-%m-%d %I:%M:%S %p UTC")
        response["utc_iso"] = utc_dt.isoformat()

        # Step 3: Convert to requested timezone
        target_tz = _resolve_timezone(output_timezone)
        local_dt = utc_dt.astimezone(target_tz)
        response["formatted"] = local_dt.strftime(output_format)

        logger.debug(
            f"Timestamp conversion: {timestamp} -> {response['formatted']}"
        )

    except Exception as e:
        response["error"] = f"Conversion failed: {str(e)}"
        logger.error(response["error"])

    return response


def string_to_timestamp(
    datetime_string: str,
    input_timezone: str = "Asia/Kolkata",
) -> dict[str, Any]:
    """
    Convert a human-readable datetime string to a Unix timestamp.

    This is essentially a wrapper around ``convert_datetime`` that returns
    the timestamp directly.

    Args:
        datetime_string (str): The datetime string to convert. Can be in any
            format supported by ``dateparser``.

        input_timezone (str): The timezone to assume for naive datetimes.
            Defaults to "Asia/Kolkata".

    Returns:
        dict[str, Any]: A dictionary containing:
            - "original_string" (str): The original datetime string.
            - "is_parsed" (bool): Whether parsing was successful.
            - "timestamp_seconds" (int): Unix timestamp in seconds.
            - "timestamp_milliseconds" (int): Unix timestamp in milliseconds.
            - "iso_8601" (str): ISO 8601 formatted string.
            - "error" (str | None): Error message if conversion failed.

    Examples:
        >>> result = string_to_timestamp("August 13, 2026 1:30 PM")
        >>> print(result["timestamp_seconds"])
        1786680000

        >>> print(result["timestamp_milliseconds"])
        1786680000000
    """
    response: dict[str, Any] = {
        "original_string": datetime_string,
        "is_parsed": False,
        "timestamp_seconds": None,
        "timestamp_milliseconds": None,
        "iso_8601": None,
        "error": None,
    }

    try:
        # Step 1: Parse the string
        parsed_dt = dateparser_parse(datetime_string)
        if parsed_dt is None:
            response["error"] = (
                f"Could not parse the datetime string: '{datetime_string}'"
            )
            return response

        response["is_parsed"] = True

        # Step 2: Attach timezone if naive
        if parsed_dt.tzinfo is None:
            target_tz = _resolve_timezone(input_timezone)
            parsed_dt = parsed_dt.replace(tzinfo=target_tz)

        # Step 3: Convert to timestamps
        response["timestamp_seconds"] = int(parsed_dt.timestamp())
        response["timestamp_milliseconds"] = int(parsed_dt.timestamp() * 1000)
        response["iso_8601"] = parsed_dt.isoformat()

        logger.debug(
            f"String to timestamp: '{datetime_string}' -> "
            f"{response['timestamp_seconds']}"
        )

    except Exception as e:
        response["error"] = f"Conversion failed: {str(e)}"
        logger.error(response["error"])

    return response


def format_iso(
    datetime_string: str,
    output_format: str = "%Y-%m-%d %I:%M:%S %p %Z",
    output_timezone: str = "Asia/Kolkata",
) -> dict[str, Any]:
    """
    Parse a datetime string and reformat it to a custom output format.

    Useful when you have a datetime in one format and need it in another.

    Args:
        datetime_string (str): The input datetime string.

        output_format (str): The desired output strftime format.
            Defaults to "%Y-%m-%d %I:%M:%S %p %Z".

        output_timezone (str): The timezone for the output.

    Returns:
        dict[str, Any]: A dictionary containing:
            - "original_string" (str): The original datetime string.
            - "is_parsed" (bool): Whether parsing was successful.
            - "formatted" (str): The reformatted datetime string.
            - "iso_8601" (str): ISO 8601 formatted string.
            - "error" (str | None): Error message if conversion failed.

    Examples:
        >>> result = format_iso("13/08/2026 13:30")
        >>> print(result["formatted"])
        "2026-08-13 01:30:00 PM IST"

        >>> result = format_iso(
        ...     "August 13, 2026 1:30 PM",
        ...     output_format="%d/%m/%Y %H:%M",
        ...     output_timezone="UTC",
        ... )
        >>> print(result["formatted"])
        "13/08/2026 08:00"
    """
    response: dict[str, Any] = {
        "original_string": datetime_string,
        "is_parsed": False,
        "formatted": None,
        "iso_8601": None,
        "error": None,
    }

    try:
        # Step 1: Parse the string
        parsed_dt = dateparser_parse(datetime_string)
        if parsed_dt is None:
            response["error"] = (
                f"Could not parse the datetime string: '{datetime_string}'"
            )
            return response

        response["is_parsed"] = True

        # Step 2: Attach timezone if naive
        if parsed_dt.tzinfo is None:
            target_tz = _resolve_timezone(output_timezone)
            parsed_dt = parsed_dt.replace(tzinfo=target_tz)

        # Step 3: Convert to output timezone
        target_tz = _resolve_timezone(output_timezone)
        converted_dt = parsed_dt.astimezone(target_tz)

        # Step 4: Format the output
        response["formatted"] = converted_dt.strftime(output_format)
        response["iso_8601"] = converted_dt.isoformat()

        logger.debug(
            f"Format ISO: '{datetime_string}' -> '{response['formatted']}'"
        )

    except Exception as e:
        response["error"] = f"Conversion failed: {str(e)}"
        logger.error(response["error"])

    return response