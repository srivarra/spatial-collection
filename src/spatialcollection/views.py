from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, KeysView, Mapping
from typing import TYPE_CHECKING, Any

import emoji
import more_itertools as mit
import natsort as ns
import rich
import rich.console
from rich.tree import Tree

if TYPE_CHECKING:
    from .spatialcollection import SpatialCollection


class ComponentView(Mapping[str, Any]):
    """Provides a view of the components within a single SpatialData object, primarily for repr."""

    def __init__(self, component_accessor: Mapping[str, Any], sdata_key: str, component_type: str):
        self._accessor = component_accessor
        self._sdata_key = sdata_key
        self._component_type = component_type

    def __getitem__(self, key: str) -> Any:
        return self._accessor[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._accessor)

    def __len__(self) -> int:
        return len(self._accessor)

    def keys(self) -> KeysView[str]:
        """Return the keys of the component."""
        return self._accessor.keys()

    def __repr__(self) -> str:
        """String representation using rich.tree."""
        component_keys = ns.natsorted(list(self.keys()))
        key_count = len(component_keys)
        header = f"Component View: {self._component_type.capitalize()} for '{self._sdata_key}'"

        tree = Tree(header)
        if key_count > 0:
            tree.add(f"Keys ({key_count}): {component_keys}")
        else:
            tree.add(f"(No {self._component_type} components found for '{self._sdata_key}')")

        # Render the tree using a temporary console
        console = rich.console.Console()
        with console.capture() as capture:
            console.print(tree)
        return capture.get().strip()


class SpatialComponentView(Mapping[str, Any]):
    """Helper class to access a specific component type across the SpatialCollection."""

    def __init__(self, collection: SpatialCollection, component_type: str):
        self._collection = collection
        self._component_type = component_type

    def __getitem__(self, key: str) -> ComponentView:
        """Get a ComponentView for a specific SpatialData key."""
        sdata = self._collection.sdatas[key]  # Raises KeyError if key not found
        if not hasattr(sdata, self._component_type):
            raise AttributeError(f"SpatialData object '{key}' does not have component type '{self._component_type}'.")
        component_accessor = getattr(sdata, self._component_type)
        # Wrap the accessor in ComponentView
        return ComponentView(component_accessor, key, self._component_type)

    def keys(self) -> KeysView[str]:
        """Return the keys of the SpatialData objects in the collection."""
        return self._collection.sdatas.keys()

    def __iter__(self) -> Iterator[str]:
        """Iterate over the SpatialData keys."""
        return iter(self.keys())

    def __len__(self) -> int:
        """Return the number of SpatialData objects in the collection."""
        return len(self._collection.sdatas)

    def __repr__(self) -> str:
        """String representation using a rich.tree for the grouped structure."""
        sdata_keys = list(self.keys())
        count = len(sdata_keys)
        header = f"SpatialCollection Component View for {self._component_type.capitalize()} (across {count} SpatialData object{'s' if count != 1 else ''})"

        tree = Tree(header)

        if not sdata_keys:
            tree.add("(No SpatialData objects in this collection)")
        else:
            groups = defaultdict(list)
            errors = []

            for key in sdata_keys:
                try:
                    # Temporarily get the raw accessor to check keys for grouping
                    raw_accessor = getattr(self._collection.sdatas[key], self._component_type)
                    if isinstance(raw_accessor, Mapping):
                        component_keys = tuple(ns.natsorted(list(raw_accessor.keys())))
                        group_key = component_keys if component_keys else "<Empty>"
                    else:
                        group_key = f"<Error: Accessor for {key} not Mapping ({type(raw_accessor)})>"
                        errors.append(f"{key}: Accessor not Mapping ({type(raw_accessor)}) ")
                except KeyError:
                    group_key = f"<Error: Key {key} not found>"
                    errors.append(f"{key}: Key not found")
                except AttributeError:
                    group_key = "<Missing>"
                except (TypeError, ValueError) as e:
                    group_key = f"<Error: {type(e).__name__} accessing {key}>"
                    errors.append(f"{key}: {e}")

                groups[group_key].append(key)

            all_valid_key_lists = [k for k in groups if isinstance(k, tuple)]
            all_valid_keys_identical = False
            if all_valid_key_lists:
                first_valid_key_list = all_valid_key_lists[0]
                if all(k == first_valid_key_list for k in all_valid_key_lists):
                    any(isinstance(k, str) and k.startswith("<Error:") for k in groups)
                    if len(groups) == 1 and isinstance(first_valid_key_list, tuple):
                        all_valid_keys_identical = True

            if all_valid_keys_identical:
                first_group_key_tuple = mit.first(groups.keys())
                group_sdata_keys = groups[first_group_key_tuple]
                component_type_str = self._component_type.capitalize()
                tree.add(f"SpatialData Keys ({len(group_sdata_keys)}): {ns.natsorted(group_sdata_keys)}")
                tree.add(
                    f"Common {component_type_str} Keys ({len(first_group_key_tuple)}): {list(first_group_key_tuple)}"
                )
            else:

                def sort_key(item):
                    key, _ = item
                    if isinstance(key, tuple):
                        return (0, key)
                    if key == "<Empty>":
                        return (1, key)
                    if key == "<Missing>":
                        return (2, key)
                    return (3, key)

                group_items = sorted(groups.items(), key=sort_key)
                component_emojis = {
                    "images": emoji.emojize(":framed_picture:"),
                    "points": emoji.emojize(":pushpin:"),
                    "labels": emoji.emojize(":label:"),
                    "shapes": emoji.emojize(":triangular_ruler:"),
                    "tables": emoji.emojize(":bar_chart:"),
                }
                emoji_icon = (
                    component_emojis.get(self._component_type, "") + " "
                    if self._component_type in component_emojis
                    else ""
                )

                for group_key, group_sdata_keys in group_items:
                    num_sdatas_in_group = len(group_sdata_keys)
                    sdatas_repr = ns.natsorted(group_sdata_keys)
                    if isinstance(group_key, tuple):
                        component_key_list = list(group_key)
                        num_comp_keys = len(component_key_list)
                        label = f"{emoji_icon}({num_sdatas_in_group} objects: {sdatas_repr}): Component Keys ({num_comp_keys}): {component_key_list}"
                    elif group_key == "<Missing>":
                        label = f"({num_sdatas_in_group} objects: {sdatas_repr}): <Component type '{self._component_type}' not available>"
                    elif group_key == "<Empty>":
                        label = f"{emoji_icon}({num_sdatas_in_group} objects: {sdatas_repr}): <Component type '{self._component_type}' present but empty>"
                    else:
                        label = f"({num_sdatas_in_group} objects: {sdatas_repr}): {group_key}"
                    tree.add(label)

        console = rich.console.Console()
        with console.capture() as capture:
            console.print(tree)
        output = capture.get().strip()
        if errors:
            output += "\n\nErrors encountered during repr generation:\n" + "\n".join([f"  - {e}" for e in errors])
        return output
