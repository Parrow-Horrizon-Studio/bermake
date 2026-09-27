#include <gtest/gtest.h>

#include <regex>

#include "bermake/version.h"

TEST(VersionTest, ReturnsNonEmptyString) {
    EXPECT_FALSE(bermake::version().empty());
}

TEST(VersionTest, MatchesSemverPattern) {
    const std::string v = bermake::version();
    const std::regex semver_pattern{R"(^\d+\.\d+\.\d+$)"};
    EXPECT_TRUE(std::regex_match(v, semver_pattern)) << "Expected MAJOR.MINOR.PATCH, got: " << v;
}
