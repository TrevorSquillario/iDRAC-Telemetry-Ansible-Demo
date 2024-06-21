## Update CHANGELOG.md
```
## [x.x.x]() - YYYY-MM-DD
### Added
- Added new commandlet

### Changed
- Changed existing commandlet

### Fixed
- Fixed issue on existing commandlet
```

## Bump Version
```
```

## Commit Changes
```
git commit -m 'Release 2.3.1'
git checkout main
git merge devel

# Only tag releases as this triggers a git workflow (.github/workflows/create-release.yml)
git tag v2.3.1
git push origin v2.3.1
git push origin main
```