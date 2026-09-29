import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useRef, useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { Callout, Field, IconButton, MenuButton, MenuItem, Popover, SegmentedControl, Select, Switch, Tabs } from "./index";

function TabsHarness({ onChange = vi.fn() }: { onChange?: (value: string) => void }): React.ReactNode {
  const [value, setValue] = useState("one");
  return (
    <Tabs
      idPrefix="test"
      items={[
        { id: "one", label: "One" },
        { id: "two", label: "Two", disabled: true, disabledReason: "Two needs a capability" },
        { id: "three", label: "Three" },
      ]}
      label="Sections"
      onChange={(next) => { setValue(next); onChange(next); }}
      value={value}
    />
  );
}

describe("Tabs", () => {
  it("uses a roving tabindex and moves with arrow, Home, and End keys", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<TabsHarness onChange={onChange} />);

    const [one, two, three] = screen.getAllByRole("tab");
    expect(one).toHaveAttribute("aria-selected", "true");
    expect(one).toHaveAttribute("tabindex", "0");
    expect(three).toHaveAttribute("tabindex", "-1");

    one!.focus();
    await user.keyboard("{ArrowRight}");
    expect(two).toHaveFocus();
    expect(onChange).not.toHaveBeenCalled();
    expect(two).toHaveAttribute("aria-disabled", "true");
    expect(two).toHaveAttribute("title", "Two needs a capability");

    await user.keyboard("{ArrowRight}");
    expect(three).toHaveFocus();
    expect(onChange).toHaveBeenLastCalledWith("three");
    expect(three).toHaveAttribute("aria-selected", "true");

    await user.keyboard("{ArrowRight}");
    expect(one).toHaveFocus();
    await user.keyboard("{End}");
    expect(three).toHaveFocus();
    await user.keyboard("{Home}");
    expect(one).toHaveFocus();
    expect(onChange).toHaveBeenLastCalledWith("one");
  });

  it("does not activate a disabled tab on click", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<TabsHarness onChange={onChange} />);
    await user.click(screen.getByRole("tab", { name: "Two" }));
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByRole("tab", { name: "One" })).toHaveAttribute("aria-selected", "true");
  });
});

describe("SegmentedControl", () => {
  it("exposes a radio group and selects with arrow keys", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<SegmentedControl label="Window" onChange={onChange} options={[{ value: "a", label: "A" }, { value: "b", label: "B" }]} value="a" />);
    expect(screen.getByRole("radiogroup", { name: "Window" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "A" })).toBeChecked();
    await user.click(screen.getByRole("radio", { name: "A" }));
    await user.keyboard("{ArrowRight}");
    expect(onChange).toHaveBeenLastCalledWith("b");
  });
});

describe("Switch", () => {
  it("is a native checkbox exposed with the switch role", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Switch description="Shows token boundaries" label="Nerd Mode" onChange={onChange} />);
    const control = screen.getByRole("switch", { name: /Nerd Mode/ });
    expect(control).not.toBeChecked();
    control.focus();
    await user.keyboard(" ");
    expect(control).toBeChecked();
    expect(onChange).toHaveBeenCalledOnce();
  });
});

describe("IconButton", () => {
  it("requires a label that becomes both its accessible name and hover hint", () => {
    render(<IconButton icon={<svg />} label="Close inspector" />);
    const button = screen.getByRole("button", { name: "Close inspector" });
    expect(button).toHaveAttribute("title", "Close inspector");
  });

  it("lets a disabled reason override the hover hint without changing the name", () => {
    render(<IconButton disabled icon={<svg />} label="Refresh" title="Start the backend first" />);
    expect(screen.getByRole("button", { name: "Refresh" })).toHaveAttribute("title", "Start the backend first");
  });
});

function PopoverHarness(): React.ReactNode {
  const [open, setOpen] = useState(false);
  const anchor = useRef<HTMLButtonElement>(null);
  return (
    <div>
      <button onClick={() => setOpen((value) => !value)} ref={anchor} type="button">Open settings</button>
      <button type="button">Elsewhere</button>
      <Popover anchorRef={anchor} label="Settings panel" onClose={() => setOpen(false)} open={open}><button type="button">Inside</button></Popover>
    </div>
  );
}

describe("Popover", () => {
  it("closes on Escape and returns focus to its anchor", async () => {
    const user = userEvent.setup();
    render(<PopoverHarness />);
    await user.click(screen.getByRole("button", { name: "Open settings" }));
    await user.click(screen.getByRole("button", { name: "Inside" }));
    expect(screen.getByLabelText("Settings panel")).toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(screen.queryByLabelText("Settings panel")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open settings" })).toHaveFocus();
  });

  it("closes on a press outside but not on presses inside or on the anchor", async () => {
    const user = userEvent.setup();
    render(<PopoverHarness />);
    await user.click(screen.getByRole("button", { name: "Open settings" }));
    await user.click(screen.getByRole("button", { name: "Inside" }));
    expect(screen.getByLabelText("Settings panel")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Elsewhere" }));
    expect(screen.queryByLabelText("Settings panel")).not.toBeInTheDocument();
  });
});

describe("MenuButton", () => {
  it("opens a menu, focuses the first item, and closes after a selection", async () => {
    const user = userEvent.setup();
    const onRename = vi.fn();
    render(<MenuButton icon={<svg />} label="Chat actions"><MenuItem onSelect={onRename}>Rename</MenuItem><MenuItem onSelect={vi.fn()}>Delete</MenuItem></MenuButton>);
    const trigger = screen.getByRole("button", { name: "Chat actions" });
    await user.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("menuitem", { name: "Rename" })).toHaveFocus();
    await user.keyboard("{ArrowDown}");
    expect(screen.getByRole("menuitem", { name: "Delete" })).toHaveFocus();
    await user.keyboard("{ArrowUp}{Enter}");
    expect(onRename).toHaveBeenCalledOnce();
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });
});

describe("Field", () => {
  it("labels its control and describes it with the hint and error", () => {
    render(<Field error="Pick another device" hint="CUDA is unavailable" label="Device"><Select><option>CPU</option></Select></Field>);
    const select = screen.getByRole("combobox", { name: "Device" });
    expect(select).toHaveAccessibleDescription("CUDA is unavailable Pick another device");
    expect(select).toHaveAttribute("aria-invalid", "true");
  });
});

describe("Callout", () => {
  it("announces danger as an alert and other tones as notes", () => {
    render(<><Callout tone="danger">Failed</Callout><Callout tone="warning">Careful</Callout></>);
    expect(screen.getByRole("alert")).toHaveTextContent("Failed");
    expect(screen.getByRole("note")).toHaveTextContent("Careful");
  });
});
