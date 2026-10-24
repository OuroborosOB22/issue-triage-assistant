from breakcast import features, versions


def test_major_minor_patch_are_told_apart():
    assert versions.change_kind(versions.parse("1.2.3"), versions.parse("2.0.0")) == "major"
    assert versions.change_kind(versions.parse("1.2.3"), versions.parse("1.3.0")) == "minor"
    assert versions.change_kind(versions.parse("1.2.3"), versions.parse("1.2.9")) == "patch"


def test_unparsable_versions_are_marked_not_guessed():
    assert versions.change_kind(versions.parse("latest"), versions.parse("2.0")) == "unparsed"
    extracted, missing = features.extract("latest", "2.0")
    assert extracted["major_delta"] is None
    assert "major_delta" in missing


def test_prerelease_and_downgrade_flags():
    assert versions.is_prerelease(versions.parse("2.0.0-beta1"))
    assert versions.is_prerelease(versions.parse("1.0.0.RC2"))
    assert not versions.is_prerelease(versions.parse("2.0.0"))
    assert versions.is_downgrade(versions.parse("2.0.0"), versions.parse("1.9.9"))


def test_missing_context_is_listed_not_replaced_by_a_value():
    extracted, missing = features.extract("1.0.0", "2.0.0")
    assert extracted["scope"] is None and "scope" in missing
    assert extracted["major_delta"] == 1


def test_unknown_scope_becomes_other():
    extracted, _ = features.extract("1.0.0", "1.0.1", scope="system", section="dependencies", file_type="JAR")
    assert extracted["scope"] == "other"


def test_no_outcome_or_identity_fields_are_features():
    forbidden = {"failure_category", "project", "organisation", "author", "title", "exit_code", "verdict", "test_failures"}
    assert not forbidden & set(features.ALL)
