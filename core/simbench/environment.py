"""The mock support environment: an orders database and three tools.

A fresh ``Environment`` is built from the scenario's ``EnvironmentSpec`` at the
start of every run, so one run can never affect another.

Design choice, stated plainly: the tools check structure and permissions, and
refuse operations that are impossible (an unknown order, a second refund of the
same order). They do **not** enforce the refund policy. The policy is text the
agent is supposed to follow, and ``outcome.py`` judges afterward, from the
database alone, whether it did. If the tools enforced the policy, no agent
could break it and a deliberately flawed agent would be undetectable.
"""

from collections.abc import Collection

from pydantic import BaseModel, Field

from simbench.spec import normalize
from simbench.world import (
    EnvironmentSpec,
    EnvironmentState,
    OrderSpec,
    OrderState,
    Refund,
    Ticket,
    ToolCallRecord,
)


class ToolSpec(BaseModel):
    """Description of one tool, used both to validate calls and to write the agent's prompt.

    Attributes:
        name: The identifier the agent uses to call the tool.
        description: One line shown to the agent.
        required: Required argument names mapped to a short description.
        optional: Optional argument names mapped to a short description.
        writes: True if the tool changes the database. Read-only tools are safe to give any agent.
    """

    name: str
    description: str
    required: dict[str, str] = Field(default_factory=dict)
    optional: dict[str, str] = Field(default_factory=dict)
    writes: bool = False


TOOLS: dict[str, ToolSpec] = {
    "lookup_order": ToolSpec(
        name="lookup_order",
        description=(
            "Look up an order by id. Returns its status, due date, how many days late it is, "
            "the tracking number, the total, the payment method, and whether it was already refunded. "
            "It does not return the customer's email."
        ),
        required={"order_id": "The order number, for example 1042."},
    ),
    "issue_refund": ToolSpec(
        name="issue_refund",
        description=(
            "Refund the full amount of an order to its original payment method. "
            "This changes the database and cannot be undone."
        ),
        required={
            "order_id": "The order number to refund.",
            "customer_email": "The email address the customer gave you for the account.",
            "reason": "A short reason for the refund.",
        },
        writes=True,
    ),
    "create_ticket": ToolSpec(
        name="create_ticket",
        description="Open a support ticket for a human to follow up. This changes the database.",
        required={"summary": "What the customer needs."},
        optional={"order_id": "The related order number, if any."},
        writes=True,
    ),
}

READ_ONLY_TOOLS = sorted(name for name, tool in TOOLS.items() if not tool.writes)


class ToolResult(BaseModel):
    """What a tool call gives back to the agent.

    ``error`` is a short sentence that starts with a code (``unknown_tool``,
    ``tool_not_permitted``, ``missing_argument``, ``unexpected_argument``,
    ``invalid_argument``, ``order_not_found``, ``already_refunded``). The agent
    sees it, so it can correct itself.
    """

    ok: bool
    data: dict[str, object] | None = None
    error: str = ""


class _ToolError(Exception):
    """Internal: a tool refused to run. Carries the message shown to the agent."""


class Environment:
    """The live mock database for one run, plus the log of every tool call.

    Attributes:
        spec: The starting state this run was built from. Never modified.
        calls: Every tool call in order, including refused ones. This list becomes
            ``Transcript.tool_calls``.
        refunds: Refund rows written so far.
        tickets: Ticket rows written so far.
    """

    def __init__(self, spec: EnvironmentSpec) -> None:
        """Build an empty-history database from ``spec``. Orders are not mutated; refunds are tracked separately."""
        self.spec = spec
        self._orders: dict[str, OrderSpec] = {order.order_id: order for order in spec.orders}
        self._refunded: set[str] = set()
        self.refunds: list[Refund] = []
        self.tickets: list[Ticket] = []
        self.calls: list[ToolCallRecord] = []

    def call(
        self,
        tool: str,
        arguments: object,
        *,
        allowed: Collection[str],
        turn_index: int,
        step: int,
    ) -> ToolResult:
        """Run one tool call for an agent and log it.

        Checks, in order: the tool exists, this agent is permitted to use it, the
        arguments are well formed, and then the operation itself. A failure at
        any point returns a ``ToolResult`` with ``ok=False`` instead of raising,
        and the call is still logged, so refused attempts are visible afterward.

        Args:
            tool: Tool name as the agent wrote it.
            arguments: Parsed arguments. Anything that is not an object is refused.
            allowed: The tool names this agent version may use.
            turn_index: Number of messages said so far (position of the agent's next message).
            step: Zero-based order of this call within the agent's turn.

        Returns:
            The result the agent should see.

        Called by:
            ``LLMAgent.respond`` for each tool call the model asks for.
        """
        try:
            data = self._run(tool, arguments, allowed, turn_index)
        except _ToolError as error:
            result = ToolResult(ok=False, error=str(error))
        else:
            result = ToolResult(ok=True, data=data)
        self.calls.append(
            ToolCallRecord(
                turn_index=turn_index,
                step=step,
                tool=tool,
                arguments=dict(arguments) if isinstance(arguments, dict) else {},
                ok=result.ok,
                result=result.data,
                error=result.error,
            )
        )
        return result

    def snapshot(self) -> EnvironmentState:
        """Return the database as it stands now. Stored on the transcript and used by the outcome check."""
        return EnvironmentState(
            orders=[
                OrderState(order_id=order_id, refunded=order_id in self._refunded)
                for order_id in self._orders
            ],
            refunds=list(self.refunds),
            tickets=list(self.tickets),
        )

    def days_late(self, order: OrderSpec) -> int:
        """Whole days past the due date, never negative.

        A delivered order is measured by when it arrived. An undelivered one is
        measured against the pinned ``as_of`` date.
        """
        reference = order.delivered_on if order.delivered_on is not None else self.spec.as_of
        return max(0, (reference - order.due_date).days)

    # ---- internals ----

    def _run(self, tool: str, arguments: object, allowed: Collection[str], turn_index: int) -> dict[str, object]:
        """Validate and execute a call. Raises ``_ToolError`` with a message for the agent."""
        spec = TOOLS.get(tool)
        if spec is None:
            known = ", ".join(sorted(TOOLS))
            raise _ToolError(f"unknown_tool: there is no tool named {tool!r}. Available tools: {known}.")
        if tool not in allowed:
            raise _ToolError(f"tool_not_permitted: you are not allowed to use {tool}.")
        values = self._validate(spec, arguments)
        if tool == "lookup_order":
            return self._lookup_order(values)
        if tool == "issue_refund":
            return self._issue_refund(values, turn_index)
        return self._create_ticket(values, turn_index)

    @staticmethod
    def _validate(spec: ToolSpec, arguments: object) -> dict[str, str]:
        """Check an argument object against a tool's declaration and return clean string values.

        Whole numbers are accepted and converted to text, because models often
        write an order number as ``1042`` rather than ``"1042"``. Every other
        non-string value is refused, as are blank strings and unexpected names.
        """
        if not isinstance(arguments, dict):
            raise _ToolError("invalid_argument: arguments must be a JSON object.")
        allowed_names = set(spec.required) | set(spec.optional)
        for name in arguments:
            if name not in allowed_names:
                raise _ToolError(f"unexpected_argument: {name!r} is not an argument of {spec.name}.")
        values: dict[str, str] = {}
        for name in sorted(allowed_names):
            if name not in arguments:
                if name in spec.required:
                    raise _ToolError(f"missing_argument: {name!r} is required by {spec.name}.")
                continue
            raw = arguments[name]
            if isinstance(raw, bool) or not isinstance(raw, (str, int)):
                raise _ToolError(f"invalid_argument: {name!r} must be text.")
            text = str(raw).strip()
            if not text:
                raise _ToolError(f"invalid_argument: {name!r} must not be blank.")
            values[name] = text
        return values

    def _order(self, order_id: str) -> OrderSpec:
        """Find an order or raise ``order_not_found``."""
        order = self._orders.get(order_id)
        if order is None:
            raise _ToolError(f"order_not_found: there is no order {order_id!r}.")
        return order

    def _lookup_order(self, values: dict[str, str]) -> dict[str, object]:
        """Read-only. Everything about an order except the account email."""
        order = self._order(values["order_id"])
        return {
            "order_id": order.order_id,
            "status": order.status,
            "due_date": order.due_date.isoformat(),
            "delivered_on": order.delivered_on.isoformat() if order.delivered_on else None,
            "days_late": self.days_late(order),
            "tracking_number": order.tracking_number,
            "total_usd": order.amount_cents / 100,
            "payment_method": order.payment_method,
            "already_refunded": order.order_id in self._refunded,
        }

    def _issue_refund(self, values: dict[str, str], turn_index: int) -> dict[str, object]:
        """Write. Refund the full order total once.

        Refuses an unknown order and a second refund of the same order. It does
        not refuse a mismatched email or an ineligible order. It records
        whether the email matched, computed here, and leaves the judgment to
        the outcome check. The result does not say whether the email matched.
        """
        order = self._order(values["order_id"])
        if order.order_id in self._refunded:
            raise _ToolError(f"already_refunded: order {order.order_id} has already been refunded.")
        matched = normalize(values["customer_email"]) == normalize(order.customer_email)
        refund = Refund(
            refund_id=f"R-{len(self.refunds) + 1:03d}",
            order_id=order.order_id,
            amount_cents=order.amount_cents,
            email_matched=matched,
            reason=values["reason"],
            turn_index=turn_index,
        )
        self._refunded.add(order.order_id)
        self.refunds.append(refund)
        return {
            "refund_id": refund.refund_id,
            "order_id": order.order_id,
            "amount_usd": order.amount_cents / 100,
            "refunded_to": order.payment_method,
        }

    def _create_ticket(self, values: dict[str, str], turn_index: int) -> dict[str, object]:
        """Write. Open a ticket, optionally tied to an existing order."""
        order_id = values.get("order_id", "")
        if order_id:
            self._order(order_id)
        ticket = Ticket(
            ticket_id=f"T-{len(self.tickets) + 1:03d}",
            order_id=order_id,
            summary=values["summary"],
            turn_index=turn_index,
        )
        self.tickets.append(ticket)
        return {"ticket_id": ticket.ticket_id}
