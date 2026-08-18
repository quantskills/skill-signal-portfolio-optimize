# Source Boundary

Keep source code, small synthetic fixtures, configuration examples, and contracts in this repository.

Keep the following outside the repository:

- API credentials, tokens, account identifiers, and private endpoints
- Raw or processed proprietary market and fundamental data
- Large model predictions, covariance matrices, holdings, and benchmark histories
- Real experiment outputs, logs, checkpoints, and reports

The runtime reads user-authorized local files only. It does not download data, call a broker, or submit orders.

When adapting another project, verify its license first. Reimplement behavior from documented contracts when code reuse would create unwanted licensing or ownership obligations.
