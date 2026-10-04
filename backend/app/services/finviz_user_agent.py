"""Send Finviz a current browser User-Agent.

finvizfinance hard-codes a 2020 Chrome/81 User-Agent, which Finviz has answered
with HTTP 403 since late September 2026. Every finvizfinance request reads the
shared ``util.headers`` dict (``finvizfinance.quote`` imports it by name), so
it is updated in place. Import this module wherever finvizfinance is used.
"""

from finvizfinance import util

# ponytail: one pinned browser string; bump it if Finviz starts refusing it too.
FINVIZ_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)

util.headers["User-Agent"] = FINVIZ_USER_AGENT
