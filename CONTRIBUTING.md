Contributing Guide

Thank you for your interest in contributing to MRI-Preprocessing-Final! Contributions of all kinds are welcome: bug reports, feature requests, documentation improvements, and pull requests.

How to Contribute
1. Fork the repository and create a feature branch from main.
2. Make your changes in small, focused commits.
3. If you add or change behavior, update or add documentation in the README when appropriate.
4. Ensure scripts run from the project root and do not hardcode absolute paths.
5. Open a pull request describing the changes and motivation.

Coding Guidelines
- Python 3.9+ compatible
- Keep dependencies minimal; prefer using what’s already in environment.yml
- Use clear, descriptive variable and function names
- Add inline comments where logic is non-trivial

Testing
- Run scripts on a small subset of patients to validate the expected inputs/outputs
- Validate that file suffixes and folder structures are preserved as described

Reporting Issues
- Use GitHub Issues
- Include: environment (OS, Python), package versions, steps to reproduce, expected vs. actual behavior, and relevant logs

License
By contributing, you agree that your contributions will be licensed under the MIT License.
