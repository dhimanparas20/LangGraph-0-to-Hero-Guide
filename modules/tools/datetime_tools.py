"""Datetime tools built on modules.datetime_parser.

These give the LLM accurate time superpowers: current time in any timezone,
parsing any human datetime phrase, converting across timezones, and
measuring durations - all via dateparser, never guessed by the model.
"""

from langchain_core.tools import tool

from modules.datetime_parser import convert_datetime, string_to_timestamp


@tool
def get_current_time(timezone: str) -> str:
    """Current date and time in any timezone. Use for every 'what time is it' question.

    Args:
        timezone: a timezone alias (e.g. "IST", "PST", "UTC") or IANA name
            (e.g. "Asia/Kolkata", "America/New_York").

    Returns:
        A one-line report with the current date and time in that timezone.
    """
    result = convert_datetime(datetime_string="now", parse_to=[timezone])

    if not result["is_parsed"]:
        return "Error: could not determine the current time."

    entry = result["custom_timezones"].get(timezone, {})
    if entry.get("error"):
        return "Error: " + str(entry["error"])

    return "Current time in " + timezone + ": " + str(entry["datetime"])


@tool
def convert_time(datetime_string: str, to_timezones: str) -> str:
    """Converts any human datetime phrase into one or more timezones.
    Understands phrases like 'Tomorrow at 10 AM', '13/08/2026 13:30', '2 hours ago'.

    Args:
        datetime_string: the datetime phrase to convert.
        to_timezones: comma-separated target timezones, e.g. "IST,UTC,America/New_York".

    Returns:
        A multi-line report: the parsed datetime (and UTC) plus one line per
        requested timezone, or an error message.
    """
    timezones = [tz.strip() for tz in to_timezones.split(",") if tz.strip()]
    result = convert_datetime(datetime_string=datetime_string, parse_to=timezones, parse_to_utc=True)

    if not result["is_parsed"]:
        return "Error: " + str(result.get("error", "could not parse the datetime."))

    lines = [
        "Parsed: " + str(result["parsed_datetime"]),
        "UTC: " + str(result["utc"]),
    ]
    for tz in timezones:
        entry = result["custom_timezones"].get(tz, {})
        if entry.get("error"):
            lines.append(tz + ": error - " + str(entry["error"]))
        else:
            lines.append(tz + ": " + str(entry["datetime"]))
    return "\n".join(lines)


@tool
def time_between(start: str, end: str) -> str:
    """Duration between two human datetime phrases, e.g. time_between('now', '2026-12-31 23:59').

    Args:
        start: the start datetime phrase (any format dateparser understands).
        end: the end datetime phrase.

    Returns:
        A one-line duration report in days, hours and minutes, or an error.
    """
    start_result = string_to_timestamp(start)
    end_result = string_to_timestamp(end)

    if not start_result["is_parsed"]:
        return "Error: could not parse the start datetime: " + start
    if not end_result["is_parsed"]:
        return "Error: could not parse the end datetime: " + end

    diff_seconds = abs(end_result["timestamp_seconds"] - start_result["timestamp_seconds"])
    days = diff_seconds // 86400
    hours = (diff_seconds % 86400) // 3600
    minutes = (diff_seconds % 3600) // 60

    return ("Between '" + start + "' and '" + end + "': "
            + str(days) + " days, " + str(hours) + " hours, " + str(minutes) + " minutes")
