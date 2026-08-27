import copy
import datetime
import unittest

from tools.validate_macos_fskit_profile import (
    ProfileValidationError,
    validate_profile,
)


TEAM_ID = "2HS27B8739"
BUNDLE_ID = "ai.metricspace.greppy.workspacefs.extension"
APP_BUNDLE_ID = "ai.metricspace.greppy.workspacefs"
APPLICATION_GROUP = "group.ai.metricspace.greppy"


def valid_profile() -> dict[str, object]:
    return {
        "ExpirationDate": datetime.datetime(2099, 1, 1),
        "ProvisionsAllDevices": True,
        "Platform": ["OSX"],
        "TeamIdentifier": [TEAM_ID],
        "ApplicationIdentifierPrefix": [TEAM_ID],
        "DeveloperCertificates": [b"certificate"],
        "Entitlements": {
            "com.apple.application-identifier": f"{TEAM_ID}.{BUNDLE_ID}",
            "com.apple.developer.team-identifier": TEAM_ID,
            "com.apple.developer.fskit.fsmodule": True,
            "com.apple.security.application-groups": [APPLICATION_GROUP],
        },
    }


class FSKitProfileValidationTests(unittest.TestCase):
    def test_accepts_exact_developer_id_fskit_profile(self) -> None:
        self.assertEqual(
            validate_profile(valid_profile(), BUNDLE_ID, APPLICATION_GROUP), TEAM_ID
        )

    def test_rejects_expired_profile(self) -> None:
        profile = valid_profile()
        profile["ExpirationDate"] = datetime.datetime(2020, 1, 1)
        with self.assertRaisesRegex(ProfileValidationError, "expired"):
            validate_profile(profile, BUNDLE_ID, APPLICATION_GROUP)

    def test_rejects_development_profile(self) -> None:
        profile = valid_profile()
        profile["ProvisionsAllDevices"] = False
        with self.assertRaisesRegex(ProfileValidationError, "Developer ID distribution"):
            validate_profile(profile, BUNDLE_ID, APPLICATION_GROUP)

    def test_rejects_wrong_bundle_identifier(self) -> None:
        profile = valid_profile()
        entitlements = copy.deepcopy(profile["Entitlements"])
        assert isinstance(entitlements, dict)
        entitlements["com.apple.application-identifier"] = (
            f"{TEAM_ID}.ai.metricspace.wrong"
        )
        profile["Entitlements"] = entitlements
        with self.assertRaisesRegex(ProfileValidationError, "does not authorize"):
            validate_profile(profile, BUNDLE_ID, APPLICATION_GROUP)

    def test_rejects_missing_fskit_authorization(self) -> None:
        profile = valid_profile()
        entitlements = copy.deepcopy(profile["Entitlements"])
        assert isinstance(entitlements, dict)
        entitlements.pop("com.apple.developer.fskit.fsmodule")
        profile["Entitlements"] = entitlements
        with self.assertRaisesRegex(ProfileValidationError, "FSKit Module"):
            validate_profile(profile, BUNDLE_ID, APPLICATION_GROUP)

    def test_rejects_wrong_application_group_when_profile_restricts_groups(self) -> None:
        profile = valid_profile()
        entitlements = copy.deepcopy(profile["Entitlements"])
        assert isinstance(entitlements, dict)
        entitlements["com.apple.security.application-groups"] = ["group.invalid"]
        profile["Entitlements"] = entitlements
        with self.assertRaisesRegex(ProfileValidationError, "application group"):
            validate_profile(profile, BUNDLE_ID, APPLICATION_GROUP)

    def test_rejects_missing_application_group(self) -> None:
        profile = valid_profile()
        entitlements = copy.deepcopy(profile["Entitlements"])
        assert isinstance(entitlements, dict)
        entitlements.pop("com.apple.security.application-groups")
        profile["Entitlements"] = entitlements
        with self.assertRaisesRegex(ProfileValidationError, "application group"):
            validate_profile(profile, BUNDLE_ID, APPLICATION_GROUP)

    def test_accepts_host_app_profile_without_fskit_entitlement(self) -> None:
        profile = valid_profile()
        entitlements = copy.deepcopy(profile["Entitlements"])
        assert isinstance(entitlements, dict)
        entitlements["com.apple.application-identifier"] = (
            f"{TEAM_ID}.{APP_BUNDLE_ID}"
        )
        entitlements.pop("com.apple.developer.fskit.fsmodule")
        profile["Entitlements"] = entitlements
        self.assertEqual(
            validate_profile(
                profile,
                APP_BUNDLE_ID,
                APPLICATION_GROUP,
                require_fskit=False,
                signing_certificate_der=b"certificate",
            ),
            TEAM_ID,
        )

    def test_rejects_profile_for_another_signing_certificate(self) -> None:
        with self.assertRaisesRegex(ProfileValidationError, "signing certificate"):
            validate_profile(
                valid_profile(),
                BUNDLE_ID,
                APPLICATION_GROUP,
                signing_certificate_der=b"different-certificate",
            )


if __name__ == "__main__":
    unittest.main()
