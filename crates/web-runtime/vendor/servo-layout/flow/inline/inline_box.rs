/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::vec::IntoIter;

use app_units::Au;
use fonts::FontRef;
use layout_api::LayoutNode;
use malloc_size_of_derive::MallocSizeOf;
use script::layout_dom::ServoLayoutNode;
use servo_arc::Arc as ServoArc;
use style::Zero;
use style::computed_values::alignment_baseline::T as AlignmentBaseline;
use style::computed_values::baseline_source::T as BaselineSource;
use style::computed_values::box_decoration_break::T as BoxDecorationBreak;
use style::context::SharedStyleContext;
use style::properties::ComputedValues;
use style::values::computed::{BaselineShift, LengthPercentage};

use super::{
    InlineContainerState, InlineContainerStateFlags, SharedInlineStyles,
    inline_container_needs_strut,
};
use crate::ContainingBlock;
use crate::cell::ArcRefCell;
use crate::context::LayoutContext;
use crate::dom_traversal::NodeAndStyleInfo;
use crate::fragment_tree::BaseFragmentInfo;
use crate::layout_box_base::LayoutBoxBase;
use crate::style_ext::{ComputedValuesExt, LayoutStyle, PaddingBorderMargin};

#[derive(Debug, MallocSizeOf)]
pub(crate) struct InlineBox {
    pub base: LayoutBoxBase,
    /// The [`SharedInlineStyles`] for this [`InlineBox`] that are used to share styles
    /// with all [`super::TextRun`] children.
    pub(super) shared_inline_styles: SharedInlineStyles,
    /// The identifier of this inline box in the containing [`super::InlineFormattingContext`].
    pub(super) identifier: InlineBoxIdentifier,
    /// The index of the default font in the [`super::InlineFormattingContext`]'s font metrics store.
    /// This is initialized during IFC shaping.
    pub default_font: Option<FontRef>,
    /// Whether or not this [`InlineBox`] breaks shaping at the start.
    pub breaks_shaping_at_start: bool,
    /// Whether or not this [`InlineBox`] breaks shaping at the end.
    pub breaks_shaping_at_end: bool,
}

impl InlineBox {
    pub(crate) fn new(info: &NodeAndStyleInfo, context: &LayoutContext) -> Self {
        let (breaks_shaping_at_start, breaks_shaping_at_end) =
            inline_box_style_breaks_shaping(&info.style);
        Self {
            base: LayoutBoxBase::new(info.into(), info.style.clone()),
            shared_inline_styles: SharedInlineStyles::from_info_and_context(info, context),
            // This will be assigned later, when the box is actually added to the IFC.
            identifier: InlineBoxIdentifier::default(),
            default_font: None,
            breaks_shaping_at_start,
            breaks_shaping_at_end,
        }
    }

    #[inline]
    pub(crate) fn layout_style(&self) -> LayoutStyle<'_> {
        LayoutStyle::Default(&self.base.style)
    }

    pub(crate) fn repair_style(
        &mut self,
        context: &SharedStyleContext,
        node: &ServoLayoutNode,
        new_style: &ServoArc<ComputedValues>,
    ) {
        (self.breaks_shaping_at_start, self.breaks_shaping_at_end) =
            inline_box_style_breaks_shaping(new_style);

        self.base.repair_style(new_style);
        *self.shared_inline_styles.style.borrow_mut() = new_style.clone();
        *self.shared_inline_styles.selected.borrow_mut() = node.selected_style(context);
    }
}

#[derive(Debug, Default, MallocSizeOf)]
pub(crate) struct InlineBoxes {
    /// A collection of all inline boxes in a particular [`super::InlineFormattingContext`].
    inline_boxes: Vec<ArcRefCell<InlineBox>>,

    /// A list of tokens that represent the actual tree of inline boxes, while allowing
    /// easy traversal forward and backwards through the tree. This structure is also
    /// stored in the [`super::InlineFormattingContext::inline_items`], but this version is
    /// faster to iterate.
    inline_box_tree: Vec<InlineBoxTreePathToken>,

    /// GREPPY FORK: the parent of each inline box, indexed by
    /// `index_in_inline_boxes`. Recorded during construction so [`Self::get_path`]
    /// can walk ancestors in O(depth) instead of scanning the token slice
    /// between the two boxes, which is O(distance) per query and Θ(N²) across
    /// the lines of a large document (2.6e9 token visits for 7.3e5 produced
    /// tokens on a 7.6MB page).
    parents: Vec<Option<InlineBoxIdentifier>>,

    /// GREPPY FORK: the stack of currently open inline boxes during
    /// construction, used only to fill `parents`. Empty once construction
    /// is complete.
    open_stack: Vec<InlineBoxIdentifier>,
}

impl InlineBoxes {
    pub(super) fn len(&self) -> usize {
        self.inline_boxes.len()
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &ArcRefCell<InlineBox>> {
        self.inline_boxes.iter()
    }

    pub(super) fn get(&self, identifier: &InlineBoxIdentifier) -> ArcRefCell<InlineBox> {
        self.inline_boxes[identifier.index_in_inline_boxes as usize].clone()
    }

    pub(super) fn end_inline_box(&mut self, identifier: InlineBoxIdentifier) {
        self.inline_box_tree
            .push(InlineBoxTreePathToken::End(identifier));
        // GREPPY FORK: keep the construction stack in sync for `parents`.
        if self.open_stack.last() == Some(&identifier) {
            self.open_stack.pop();
        }
    }

    pub(super) fn start_inline_box(
        &mut self,
        inline_box: ArcRefCell<InlineBox>,
    ) -> InlineBoxIdentifier {
        assert!(self.inline_boxes.len() <= u32::MAX as usize);
        assert!(self.inline_box_tree.len() <= u32::MAX as usize);

        let index_in_inline_boxes = self.inline_boxes.len() as u32;
        let index_of_start_in_tree = self.inline_box_tree.len() as u32;

        let identifier = InlineBoxIdentifier {
            index_of_start_in_tree,
            index_in_inline_boxes,
        };
        inline_box.borrow_mut().identifier = identifier;

        self.inline_boxes.push(inline_box);
        self.inline_box_tree
            .push(InlineBoxTreePathToken::Start(identifier));

        // GREPPY FORK: record the parent for O(depth) path queries.
        self.parents.push(self.open_stack.last().copied());
        self.open_stack.push(identifier);

        identifier
    }

    /// GREPPY FORK: the ancestor chain of `of`, from itself up to the root
    /// (inclusive of `of`, exclusive of the imaginary IFC root).
    fn ancestors(&self, of: InlineBoxIdentifier) -> Vec<InlineBoxIdentifier> {
        let mut chain = Vec::new();
        let mut current = Some(of);
        while let Some(identifier) = current {
            chain.push(identifier);
            current = self.parents[identifier.index_in_inline_boxes as usize];
        }
        chain
    }

    /// GREPPY FORK: O(depth) ancestor-walk path query. The released
    /// implementation ([`Self::get_path_original`]) scans the token slice
    /// between the two boxes, O(distance) per query; since every line starts
    /// its path query from the IFC root, that is Θ(N²) across a large
    /// document. This walk produces the identical token sequence: the End
    /// tokens from `from` up to (exclusive) the lowest common ancestor,
    /// then the Start tokens down to `to`.
    ///
    /// `GREPPY_GET_PATH_ORIGINAL=1` switches back to the released scan;
    /// `GREPPY_GET_PATH_VERIFY=1` runs both and panics on any divergence.
    pub(super) fn get_path(
        &self,
        from: Option<InlineBoxIdentifier>,
        to: InlineBoxIdentifier,
    ) -> IntoIter<InlineBoxTreePathToken> {
        #[derive(Clone, Copy, PartialEq)]
        enum Mode {
            Ancestors,
            Original,
            Verify,
        }
        static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
        let mode = *MODE.get_or_init(|| {
            if std::env::var_os("GREPPY_GET_PATH_VERIFY").is_some() {
                Mode::Verify
            } else if std::env::var_os("GREPPY_GET_PATH_ORIGINAL").is_some() {
                Mode::Original
            } else {
                Mode::Ancestors
            }
        });
        match mode {
            Mode::Ancestors => self.get_path_via_ancestors(from, to),
            Mode::Original => self.get_path_original(from, to),
            Mode::Verify => {
                let fast: Vec<_> = self.get_path_via_ancestors(from, to).collect();
                let slow: Vec<_> = self.get_path_original(from, to).collect();
                assert_eq!(
                    fast, slow,
                    "get_path divergence for from={from:?} to={to:?}"
                );
                fast.into_iter()
            },
        }
    }

    /// GREPPY FORK: see [`Self::get_path`].
    fn get_path_via_ancestors(
        &self,
        from: Option<InlineBoxIdentifier>,
        to: InlineBoxIdentifier,
    ) -> IntoIter<InlineBoxTreePathToken> {
        if from == Some(to) {
            return Vec::new().into_iter();
        }
        let to_chain = self.ancestors(to);
        let mut path = Vec::with_capacity(to_chain.len() + 4);
        let mut lca: Option<InlineBoxIdentifier> = None;
        if let Some(from) = from {
            let mut current = Some(from);
            while let Some(identifier) = current {
                if to_chain.contains(&identifier) {
                    lca = Some(identifier);
                    break;
                }
                path.push(InlineBoxTreePathToken::End(identifier));
                current = self.parents[identifier.index_in_inline_boxes as usize];
            }
        }
        let descend_count = match lca {
            Some(lca) => to_chain
                .iter()
                .position(|identifier| *identifier == lca)
                .expect("lca came from to_chain"),
            None => to_chain.len(),
        };
        path.extend(
            to_chain[..descend_count]
                .iter()
                .rev()
                .map(|identifier| InlineBoxTreePathToken::Start(*identifier)),
        );
        path.into_iter()
    }

    /// The released 0.5.0 implementation, kept verbatim for the
    /// `GREPPY_GET_PATH_ORIGINAL` switch and the verify mode.
    fn get_path_original(
        &self,
        from: Option<InlineBoxIdentifier>,
        to: InlineBoxIdentifier,
    ) -> IntoIter<InlineBoxTreePathToken> {
        if from == Some(to) {
            return Vec::new().into_iter();
        }

        let mut from_index = match from {
            Some(InlineBoxIdentifier {
                index_of_start_in_tree,
                ..
            }) => index_of_start_in_tree as usize,
            None => 0,
        };
        let mut to_index = to.index_of_start_in_tree as usize;
        let is_reversed = to_index < from_index;

        // Do not include the first or final token, depending on direction. These can be equal
        // if we are starting or going to the root of the inline formatting context, in which
        // case we don't want to adjust.
        if to_index > from_index && from.is_some() {
            from_index += 1;
        } else if to_index < from_index {
            to_index += 1;
        }

        let mut path = Vec::with_capacity(from_index.abs_diff(to_index));
        let min = from_index.min(to_index);
        let max = from_index.max(to_index);

        for token in &self.inline_box_tree[min..=max] {
            // Skip useless recursion into inline boxes; we are looking for a direct path.
            if Some(&token.reverse()) == path.last() {
                path.pop();
            } else {
                path.push(*token);
            }
        }

        if is_reversed {
            path.reverse();
            for token in path.iter_mut() {
                *token = token.reverse();
            }
        }

        path.into_iter()
    }
}

#[derive(Clone, Copy, Debug, MallocSizeOf, PartialEq)]
pub(super) enum InlineBoxTreePathToken {
    Start(InlineBoxIdentifier),
    End(InlineBoxIdentifier),
}

impl InlineBoxTreePathToken {
    fn reverse(&self) -> Self {
        match self {
            Self::Start(index) => Self::End(*index),
            Self::End(index) => Self::Start(*index),
        }
    }
}

/// An identifier for a particular [`InlineBox`] to be used to fetch it from an [`InlineBoxes`]
/// store of inline boxes.
///
/// [`u32`] is used for the index, in order to save space. The value refers to the token
/// in the start tree data structure which can be fetched to find the actual index of
/// of the [`InlineBox`] in [`InlineBoxes::inline_boxes`].
#[derive(Clone, Copy, Debug, Default, Eq, Hash, MallocSizeOf, PartialEq)]
pub(crate) struct InlineBoxIdentifier {
    pub index_of_start_in_tree: u32,
    pub index_in_inline_boxes: u32,
}

pub(super) struct InlineBoxContainerState {
    /// The container state common to both [`InlineBox`] and the root of the
    /// [`super::InlineFormattingContext`].
    pub base: InlineContainerState,

    /// The [`InlineBoxIdentifier`] of this inline container state. If this is the root
    /// the identifier is [`None`].
    pub identifier: InlineBoxIdentifier,

    /// The [`BaseFragmentInfo`] of the [`InlineBox`] that this state tracks.
    pub base_fragment_info: BaseFragmentInfo,

    /// The [`PaddingBorderMargin`] of the [`InlineBox`] that this state tracks.
    pub pbm: PaddingBorderMargin,
}

impl InlineBoxContainerState {
    pub(super) fn new(
        inline_box: &InlineBox,
        containing_block: &ContainingBlock,
        layout_context: &LayoutContext,
        parent_container: &InlineContainerState,
        default_font: Option<FontRef>,
    ) -> Self {
        let style = inline_box.base.style.clone();
        let pbm = inline_box
            .layout_style()
            .padding_border_margin(containing_block);

        let mut flags = InlineContainerStateFlags::empty();
        if inline_container_needs_strut(&style, layout_context, Some(&pbm)) {
            flags.insert(InlineContainerStateFlags::CREATE_STRUT);
        }

        Self {
            base: InlineContainerState::new(style, flags, Some(parent_container), default_font),
            identifier: inline_box.identifier,
            base_fragment_info: inline_box.base.base_fragment_info,
            pbm,
        }
    }

    pub(super) fn calculate_space_above_baseline(&self) -> Au {
        let font_metrics = &self.base.font_metrics;
        let (ascent, descent, line_gap) = (
            font_metrics.ascent,
            font_metrics.descent,
            font_metrics.line_gap,
        );
        let leading = line_gap - (ascent + descent);
        leading.scale_by(0.5) + ascent
    }

    pub(super) fn should_clone_pbm(&self) -> bool {
        self.base.style.get_border().box_decoration_break == BoxDecorationBreak::Clone
    }
}

/// Whether or not an inline box style breaks shaping at the start and end.
/// The return value is `(breaks_shaping_at_start, breaks_shaping_at_end)`.
///
/// From <https://drafts.csswg.org/css-text/#boundary-shaping>
/// > Text shaping must be broken at inline box boundaries when any of the
/// > following are true for any box whose boundary separates the two typographic
/// > character units:
/// > - Any of margin/border/padding separating the two typographic character units
/// >   in the inline axis is non-zero.
/// > - `vertical-align` is not `baseline`.
/// > - The boundary is a bidi isolation boundary.
fn inline_box_style_breaks_shaping(style: &ComputedValues) -> (bool, bool) {
    // Note: These steps are reordered for performance reasons and bidi isolation is handled
    // intrinsically by the fact that bidi isolation inserts bidi characters into the inline
    // formatting context text content. This leads to non-contiguous character offsets between
    // shaping queue entries.
    if style.clone_baseline_shift() != BaselineShift::zero() ||
        style.clone_baseline_source() != BaselineSource::Auto ||
        style.clone_alignment_baseline() != AlignmentBaseline::Baseline
    {
        return (true, true);
    }

    let layout_style = LayoutStyle::Default(style);
    let border_widths = layout_style.border_width(style.writing_mode);
    let padding = layout_style.padding(style.writing_mode);
    let margin = style.margin(style.writing_mode);

    (
        !border_widths.inline_start.is_zero() ||
            !padding.inline_start.is_zero() ||
            !margin
                .inline_start
                .non_auto()
                .is_none_or(LengthPercentage::is_zero),
        !border_widths.inline_end.is_zero() ||
            !padding.inline_end.is_zero() ||
            !margin
                .inline_end
                .non_auto()
                .is_none_or(LengthPercentage::is_zero),
    )
}
