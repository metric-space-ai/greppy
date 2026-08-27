#!/usr/bin/env python3
"""Validate the decoded Developer ID profile for Greppy's FSKit extension."""

from __future__ import annotations

import argparse
import datetime
import pathlib
import plistlib
from collections.abc import Mapping
from typing import Any


class ProfileValidationError(ValueError):
    """The profile cannot authorize the shipped FSKit extension."""


def _fail(message: str) -> None:
    raise ProfileValidationError(
        f"invalid FSKit Developer ID provisioning profile: {message}"
    )


def validate_profile(
    profile: Mapping[str, Any], bundle_id: str, application_group: str
) -> str:
    expiration = profile.get("ExpirationDate")
    if not isinstance(expiration, datetime.datetime):
        _fail("ExpirationDate is missing")
    if expiration.tzinfo is None:
        expiration = expiration.replace(tzinfo=datetime.timezone.utc)
    if expiration <= datetime.datetime.now(datetime.timezone.utc):
        _fail("profile has expired")
    if profile.get("ProvisionsAllDevices") is not True:
        _fail("profile is not a Developer ID distribution profile")

    platforms = profile.get("Platform")
    if not isinstance(platforms, list) or not any(
        platform in {"OSX", "macOS"} for platform in platforms
    ):
        _fail("profile does not target macOS")
    team_ids = profile.get("TeamIdentifier")
    if not isinstance(team_ids, list) or len(team_ids) != 1 or not team_ids[0]:
        _fail("profile must bind exactly one TeamIdentifier")
    team_id = team_ids[0]
    if not isinstance(team_id, str):
        _fail("TeamIdentifier is not a string")
    prefixes = profile.get("ApplicationIdentifierPrefix")
    if not isinstance(prefixes, list) or team_id not in prefixes:
        _fail("application identifier prefix does not contain the team identifier")

    entitlements = profile.get("Entitlements")
    if not isinstance(entitlements, dict):
        _fail("Entitlements dictionary is missing")
    if entitlements.get("com.apple.developer.team-identifier") != team_id:
        _fail("entitlement team identifier does not match the profile")
    expected_application_id = f"{team_id}.{bundle_id}"
    if entitlements.get("com.apple.application-identifier") != expected_application_id:
        _fail(f"profile does not authorize {bundle_id}")
    if entitlements.get("com.apple.developer.fskit.fsmodule") is not True:
        _fail("FSKit Module entitlement is not authorized")
    groups = entitlements.get("com.apple.security.application-groups")
    if groups is not None and (
        not isinstance(groups, list) or application_group not in groups
    ):
        _fail(f"profile does not authorize application group {application_group}")
    certificates = profile.get("DeveloperCertificates")
    if not isinstance(certificates, list) or not certificates:
        _fail("DeveloperCertificates is empty")
    return team_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plist", required=True, type=pathlib.Path)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--application-group", required=True)
    arguments = parser.parse_args()
    with arguments.plist.open("rb") as profile_file:
        profile = plistlib.load(profile_file)
    try:
        team_id = validate_profile(
            profile, arguments.bundle_id, arguments.application_group
        )
    except ProfileValidationError as error:
        parser.error(str(error))
    print(team_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
