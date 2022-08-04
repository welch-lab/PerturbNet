# Command-line program to view some of the syntax of a SMILES string

# Copyright Andrew Dalke, Andrew Dalke Scientific AB (Sweden), 2018
# and distributed under the "MIT license", as follows:
# ============================================================
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================

# Ideas:
#  - add closure labels on the left/right of the closure track
#  - show a 'bonds' track showing the bond types;  -, =, #, $, :, and ~
#       unknown/unspecified, as when no toolkit is available.
#  - show a 'chiralities' track with atom-ordered display and either
#     'CW' or 'CCW' as the stereo label. Test cases at
#    https://github.com/rdkit/rdkit/commit/8fdb7fb51c26a056124cccb7c3b4c610d04bfa04#diff-c1fe48c12a002a56de0af4e3296cce3eR118
#    (Suggested by Brian Cole.)
#  - use the toolkit to perceive chemsitry and set the atom/bond terms
#     (eg, if input_smiles is 'C=1-C=C-C=C-C=1' then the smiles should be
#       'c:1:c:c:c:c:c:1')

# Note: This code and the command-line API ARE NOT STABLE.
#
# The project started as an off-the-cuff idea that grew into a
# prototype and is now on its way to being a proper tool. As a result,
# it contains some half-baked ideas which are likely to change,
# perhaps even drastically, over time.

from __future__ import print_function

import re
import sys
import codecs
import inspect
from collections import defaultdict
import argparse

__version__ = "1.2"
__version_info__ = (1, 2, 0)


class ASCIISymbols:
	nw_corner = "{("  # these three alternate
	e_side = "{("
	sw_corner = "{("
	single_row = "["
	## left_closure = "%"
	## right_closure = "%"
	closure_atom = "*"
	closure_other_atoms = "x"
	closure_label = "^"
	open_branch = "("
	close_branch = ")"
	in_branch = "."

	towards_arrows = {
		"above": "V",
		"below": "^",
		}
	## secondary_atoms = {
	##     "above": "x",
	##     "below": "x",
	##     }

class UnicodeSymbols:
	nw_corner = u"\U0000250C"
	e_side = u"\U00002502"
	sw_corner = u"\U00002514"
	single_row = "["
	## left_closure = u"\U00002191"
	## right_closure = u"\U00002191"
	closure_atom = "*"
	closure_other_atoms = "x"
	closure_label = u"\U00002191"
	open_branch = "("
	close_branch = ")"
	in_branch = "."
	towards_arrows = {
		"above": u"\U00002193",
		"below": u"\U00002191",
		}
	## secondary_atoms = {
	##     "above":

# Very hacky way to handle alternating track edge indicators for
# ASCII mode.
def _get_special_symbols(symbols, counter):
	class SpecialSymbols:
		pass
	for attr in ("nw_corner", "e_side", "sw_corner", "single_row"):
		s = getattr(symbols, attr)
		n = len(s)
		c = s[counter % n]
		setattr(SpecialSymbols, attr, c)
	return SpecialSymbols


#### Various aspects of processing a SMILES, without chemistry perception.

## Set up table of valid symbols (aromatic and aliphatic), mapped to their atomic number

supported_organic_symbols = {
	'*':0,
	'B':5, 'C':6, 'N':7, 'O':8, 'S':16, 'P':15, 'F':9, 'Cl':17, 'Br':35, 'I':53,
	'b':5, 'c':6, 'n':7, 'o':8, 's':16, 'p':15,
	}

supported_bracket_symbols = [
	'*',
	'H',                                                                    'He',
	'Li', 'Be',                                                             'B',  'C',  'N',  'O',  'F',  'Ne',
	'Na', 'Mg',                                                             'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
	'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
	'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe',

	'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
					  'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',

	'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
					  'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
	]
supported_bracket_symbols = dict((symbol, i) for (i, symbol) in enumerate(supported_bracket_symbols))
supported_bracket_symbols.update({
	'b':5, 'c':6, 'n':7, 'o':8, 's':16, 'p':15, 'se':34,
	})

def verify_symbols():
	from rdkit import Chem

	def check_atom_properties(symbol, smiles, eleno, is_aromatic):
		if smiles == "[Nh]":
			assert (eleno, is_aromatic) == (113, False), (smiles, eleno, is_aromatic)
			return
		elif smiles == "[Mc]":
			assert (eleno, is_aromatic) == (115, False), (smiles, eleno, is_aromatic)
			return
		elif smiles == "[Ts]":
			assert (eleno, is_aromatic) == (117, False), (smiles, eleno, is_aromatic)
			return
		elif smiles == "[Og]":
			assert (eleno, is_aromatic) == (118, False), (smiles, eleno, is_aromatic)
			return
		mol = Chem.MolFromSmiles(smiles)
		if mol is None:
			raise AssertionError((symbol, smiles))
		a = mol.GetAtomWithIdx(0)
		if a.GetAtomicNum() != eleno:
			raise AssertionError((symbol, eleno, a.GetAtomicNum()))
		elif a.GetIsAromatic() != is_aromatic:
			raise AssertionError((symbol, is_aromatic, a.GetIsAromatic()))


	for symbol, eleno in supported_organic_symbols.items():
		if symbol in ("b", "c", "n", "p"):
			smiles = symbol + "1ccccc1"
			is_aromatic = True
		elif symbol in ("o", "s"):
			smiles = symbol + "1cccc1"
			is_aromatic = True
		else:
			smiles = symbol
			is_aromatic = False
		check_atom_properties(symbol, smiles, eleno, is_aromatic)

	for symbol, eleno in supported_bracket_symbols.items():
		bracket_symbol = "[" + symbol + "]"
		if symbol in ("b", "c", "n", "p"):
			smiles = bracket_symbol + "1ccccc1"
			is_aromatic = True
		elif symbol in ("o", "s", 'se'):
			smiles = bracket_symbol + "1cccc1"
			is_aromatic = True
		else:
			smiles = bracket_symbol
			is_aromatic = False
		check_atom_properties(symbol, smiles, eleno, is_aromatic)

#verify_symbols()

## Tokenize a SMILES string following the OpenSMILES syntax.

class Token(object):
	def __init__(self, token_index, typename, term, start, end):
		self.token_index = token_index
		self.typename = typename
		self.term = term
		self.start = start
		self.end = end
	def __repr__(self):
		return "Token(%d, %r, %r, %d, %d)" % (
			self.token_index, self.typename, self.term, self.start, self.end)

class ParseError(Exception):
	locator_msg = "Problem is here"
	def __init__(self, reason, smiles, offset, advice=None):
		self.reason = reason
		self.smiles = smiles
		self.offset = offset
		self._advice = advice
		assert 0 <= offset <= len(smiles), (offset, len(smiles))

	def __str__(self):
		return self.reason

	def get_advice(self):
		return self._advice

	def _get_highlight_messages(self):
		# This is a hook so I can reuse get_failure_highlight()
		# with ClosureError.
		return [(self.offset, self.locator_msg)]

	def get_failure_highlight(self, prefix="", width=60):
		if width < 40:
			raise ValueError("width must at least 40")
		smiles = self.smiles
		if smiles is None or self.offset is None:
			return ""
		# Break the lines up into segments
		lines = []
		for i in range(0, len(smiles), width):
			lines.append(prefix + smiles[i:i+width] + "\n")

		# Figure out where I'm going to insert the error messages.
		lines_to_add = defaultdict(list)
		for offset, errmsg in self._get_highlight_messages():
			if offset == len(smiles):
				# Need to be able to mark one past the end in case the bracket
				# atom is incomplete and the length of the SMILES is right at
				# the fold boundary.
				lineno, delta = divmod(offset-1, width)
				delta += 1
			else:
				lineno, delta = divmod(offset, width)
			# Use "XXX ^" for the message if there's enough space
			# otherwise use "^ XXX".
			if delta < (len(errmsg) + 4):
				msg = prefix + " "*delta + "^ " + errmsg + "\n"
			else:
				msg = prefix + " "*(delta - len(errmsg) - 1) + errmsg + " ^\n"
			assert lineno < len(lines)
			lines_to_add[lineno].append(msg)


		output_lines = []
		for lineno, line in enumerate(lines):
			output_lines.append(line)
			if lineno in lines_to_add:
				for insert_line in lines_to_add[lineno]:
					output_lines.append(insert_line)

		return "".join(output_lines)

	def get_report(self, msg="Cannot parse SMILES: ", prefix="  ", width=60):
		advice = self.get_advice() or ""
		if advice:
			advice += "\n"
		return (
			msg + self.reason + "\n" +
			self.get_failure_highlight(prefix=prefix) +
			advice
			)

## TokenizeError is for ParseErrors which occur in a state diagram.
# An instance can give more details about what it expects.

# Convert the states into a more readable form.
def _format_state(state):
	return state.replace("_", " ")
def _format_states(states):
	states = [_format_state(state) for state in states]
	if len(states) > 2:
		states[-2:] = ["%s, or %s" % (states[-2], states[-1])]
	return ", ".join(states)
def _format_indefinite(s):
	if s[:1] in "aeiou" or s[:6] == "hcount":
		return "n " + s
	return " " + s

class TokenizeError(ParseError):
	locator_msg = "Tokenizing stopped here"
	def __init__(self, reason, smiles, offset, state, expected_states):
		super(TokenizeError, self).__init__(reason, smiles, offset)
		self.state = state
		self.expected_states = expected_states

	def get_advice(self):
		return "A%s must be followed by a%s." % (
			_format_indefinite(_format_state(self.state)),
			_format_indefinite(_format_states(self.expected_states))
			)

class ClosureError(ParseError):
	def __init__(self, reason, smiles, offset, previous_offset, previous_locator_msg, advice=None):
		super(ClosureError, self).__init__(reason, smiles, offset, advice)
		self.previous_offset = previous_offset
		self.previous_locator_msg = previous_locator_msg

	def _get_highlight_messages(self):
		return [(self.offset, self.locator_msg),
				(self.previous_offset, self.previous_locator_msg)]

## The SMILES string tokenizer uses a regexp lexer and a state diagram


# The lexer
_smiles_lexer = re.compile(r"""
(?P<atom>   # These will be processed by 'tokenize_atom'
  \*|
  Cl|Br|[cnospBCNOFPSI]|  # organic subset
  \[[^]]+\]               # bracket atom
) |
(?P<bond>
  [=#$/\\:-]
) |
(?P<closure>
  [0-9]|          # single digit
  %[0-9][0-9]|    # two digits
  %\([0-9]+\)     # more than two digits
) |
(?P<open_branch>
  \(
) |
(?P<close_branch>
  \)
) |
(?P<dot>
  \.
)
""", re.X).match

# Allowed state transitions in an OpenSMILES grammar.
# (Some grammars are more liberal and allow things like ".C..C." and "C(.)C".)
_smiles_tokenizer_states = {
	# From state -> list of possible states
	"start": ("atom",),

	# CC, C=C, C(C)C, C(C)C, C.C, C1CCC1
	"atom": ("atom", "bond", "close_branch", "open_branch", "dot", "closure"),

	# C=C, C=1CCC=1
	"bond": ("atom", "closure"),

	# C(C)C, C(C)=C, C(C).C, C(C(C))C, C(C)(C)C
	"close_branch": ("atom", "bond", "dot", "close_branch", "open_branch"),

	# C(C), C(=C), C(.C) (really!)
	"open_branch": ("atom", "bond", "dot"),

	# C.C
	"dot": ("atom",),

	# C1CCC1, C1=CCC1, C12CC1C2, C1C(CC)1, C1(CC)CC1, c1ccccc1.[NH4+]
	"closure": ("atom", "bond", "closure", "close_branch", "open_branch", "dot"),
}


def tokenize_smiles(smiles):
	current_state = "start"
	n = len(smiles)
	start = 0
	token_index = 0
	atom_index = 0
	while start < n:
		expected = _smiles_tokenizer_states[current_state]
		m = _smiles_lexer(smiles, start)
		if m is None:
			# If it's an incomplete bracket atom then let the atom lexer report the error
			if smiles[start] == '[':
				tokenize_atom(smiles, token_index, "atom", smiles[start:], start, n, atom_index)
				raise AssertionError("How did I get here?", (smiles, smiles[start:]))
			if smiles[start] == "%" and "closure" in expected:
				next_char = smiles[start+1:start+2]
				if next_char:
					if ("0" <= next_char <= "9"):
						raise ParseError("Missing second digit after closure",
										 smiles, start)
					if next_char == '(':
						raise ParseError("Extended closures must be of the form %(N) where N is one or more digits",
										smiles, start)

			raise TokenizeError("Unexpected syntax", smiles, start, current_state, expected)

		for next_state in expected:
			term = m.group(next_state)
			if term:
				end = m.end(next_state)
				break
		else:
			raise TokenizeError("Unexpected term", smiles, start, current_state, expected)

		if next_state == "atom":
			yield tokenize_atom(smiles, token_index, next_state, term, start, end, atom_index)
			atom_index += 1
		else:
			yield Token(token_index, next_state, term, start, end)
		token_index += 1
		current_state = next_state
		start = end


# Atom tokenization is a bit tricky. There are a few cases:
#  1) it's in the organic subset, which is easy to check if the symbol is valid
#  2) it's a valid bracket atom, which can be handled through a regex
#  3) if it is not valid, fall back to a state-based tokenizer

# Options #2 and #3 share most of the regular expressions, defined here.

class AtomToken(Token):
	def __init__(self, token_index, typename, term, start, end, atom_index, symbol_start, symbol):
		super(AtomToken, self).__init__(token_index, typename, term, start, end)
		self.atom_index = atom_index
		self.symbol_start = symbol_start
		self.symbol = symbol

	@property
	def is_aromatic(self):
		return "a" <= self.symbol[:1] <= "z"

	@property
	def atomic_number(self):
		return supported_bracket_symbols[self.symbol]


class OrganicAtomToken(AtomToken):
	subtypename = "organic_atom"
	def __init__(self, token_index, typename, term, start, end, atom_index):
		super(OrganicAtomToken, self).__init__(
			token_index, typename, term, start, end, atom_index, start, term)

	def __repr__(self):
		return "OrganicAtomToken(%d, %r, %r, %d, %d, %d)" % (
			self.token_index, self.typename, self.term, self.start, self.end, self.atom_index)

	@property
	def is_heavy(self):
		return True

	def to_bracket_format(self, hcount=None):
		if hcount is None or hcount == 0:
			hcount_str = ""
		elif hcount == 1:
			hcount_str = "H"
		elif hcount > 1:
			hcount_str = "H%d" % (hcount,)
		else:
			raise ValueError("Bad hcount: %r" % (hcount,))

		new_term = "[" + self.term + hcount_str + "]"
		return new_term

class BracketAtomToken(AtomToken):
	subtypename = "bracket_atom"
	def __init__(self, token_index, typename, term, start, end, atom_index,
				 isotope, symbol_start, symbol, chiral, hcount, charge, atom_class):
		super(BracketAtomToken, self).__init__(
			token_index, typename, term, start, end, atom_index, symbol_start, symbol)
		self.isotope = isotope
		self.chiral = chiral
		self.hcount = hcount
		self.charge = charge
		self.atom_class = atom_class
		check = "[" + isotope + symbol + chiral + hcount + charge + atom_class + "]"
		assert check == term, (check, term)

	@property
	def is_heavy(self):
		if self.symbol != "H":
			return True
		if self.isotope == "":
			return False
		if int(self.isotope) in (0, 1):
			return False
		return True

	def __repr__(self):
		return ("BracketAtomToken(%d, %r, %r, %d, %d, %d, "
				 "isotope=%r, symbol_start=%d, symbol=%r, "
				 "chiral=%r, hcount=%r, charge=%r, atom_class=%r)" % (
			self.token_index, self.typename, self.term, self.start, self.end, self.atom_index,
			self.isotope, self.symbol_start, self.symbol,
			self.chiral, self.hcount, self.charge, self.atom_class))


# regular expression definitions for the different parts of a bracket atom
_start_bracket_re = r"(?P<start_bracket>\[)"
_element_re = r"(?P<isotope>\d+)"
_symbol_re = r"(?P<symbol>[A-Z][a-z]?|[a-z][a-z]?|\*)"
_chiral_re = r"(?P<chiral>@@?)"  # Not supporting the named variants like @TH1 or @OH19
_hcount_re = r"(?P<hcount>H\d*)"
_charge_re = r"(?P<charge>-(-|\d+)?|\+(\+|\d+)?)"
_atom_class_re = r"(?P<atom_class>:\d+)"
_atom_class_start_re = r"(?P<atom_class_start>:)"  # for error reporting
_atom_class_value_re = r"(?P<atom_class_value>\d+)" # for error reporting
_end_bracket_re = r"(?P<end_bracket>\])"

# Put them together to get the regular expression for a valid bracket atom.
_bracket_atom_re = (
	_start_bracket_re +
	_element_re + "?" +
	_symbol_re +
	_chiral_re + "?" +
	_hcount_re + "?" +
	_charge_re + "?" +
	_atom_class_re + "?" +
	_end_bracket_re
	)
_bracket_atom_matcher = re.compile(_bracket_atom_re + r"\Z").match

# state name -> (tokenizer, expected_states) where:
#   lexer = re.compile("|".join(regular expressions)).match
#   expected_states = group names for each of the (?P<name>...) terms
def state_options(*re_patterns):
	try:
		pat = re.compile("|".join(re_patterns))
	except Exception as err:
		# Try to pin down which regexp caused the problem
		s = str(err)
		for pat in re_patterns:
			try:
				re.compile(pat)
			except Exception as err:
				s = str(err)
				raise AssertionError(("Failed", s, pat))
		raise AssertionError(("Failed", s, "|".join(re_patterns)))
	items = sorted(pat.groupindex.items(), key=lambda item: item[1])
	expected_states = [item[0] for item in items]
	return pat.match, expected_states

# This is a complex way to write:
#   start_bracket isotope? symbol chiral? hcount? charge? atom_class? end_bracket
_bracket_atom_tokenizer_states = {
	"start": state_options(_start_bracket_re),
	"start_bracket": state_options(_element_re, _symbol_re),
	"isotope": state_options(_symbol_re),
	"symbol": state_options(_chiral_re, _hcount_re, _charge_re, _atom_class_start_re, _end_bracket_re),
	"chiral": state_options(_hcount_re, _charge_re, _atom_class_start_re, _end_bracket_re),
	"hcount": state_options(_charge_re, _atom_class_start_re, _end_bracket_re),
	"charge": state_options(_atom_class_start_re, _end_bracket_re),
	# Extra complication: I want to tokenize the atom class ":" if it exists, so I can
	# point to the missing value, if it isn't present.
	"atom_class_start": state_options(_atom_class_value_re),
	"atom_class_value": state_options(_end_bracket_re),
	"end_bracket": (None, None),
	}

def tokenize_bracket_atom(bracket_atom_term):
	return tokenize_atom(bracket_atom_term, 0, "atom", bracket_atom_term,
						 0, len(bracket_atom_term), 0)

def tokenize_atom(smiles, token_index, state, term, start, end, atom_index):
	assert state == "atom"
	if term[:1] != "[":
		# Case #1: It's in the organic subset
		if term not in supported_organic_symbols:
			raise ParseError(
				"Unsupported element symbol %r" % (term,),
				smiles, start, start+1)
		return OrganicAtomToken(token_index, state, term, start, end, atom_index)

	m = _bracket_atom_matcher(term)
	if m is not None:
		# Case #2: The complex bracket atom regexp matches.
		# Then it's a matter of picking out the right fields
		symbol = m.group("symbol")
		symbol_start = m.start("symbol")
		if symbol not in supported_bracket_symbols:
			raise ParseError(
				"Unsupported element symbol %r" % (symbol,),
				smiles, start+symbol_start, start+symbol_start+len(symbol))

		return BracketAtomToken(token_index, state, term, start, end, atom_index,
			m.group("isotope") or "",  # convert possible None values to an empty string
			start + symbol_start,
			symbol,
			m.group("chiral") or "",
			m.group("hcount") or "",
			m.group("charge") or "",
			m.group("atom_class") or "",
			)

	# Case #3: There was some sort of error.
	# Use a state table to figure out where.
	current_state = "start"
	n = len(term)
	match_start = 0
	while match_start < n:
		lexer, expected_states = _bracket_atom_tokenizer_states[current_state]
		if expected_states is None:
			# Can only happen if current_state == "end_bracket", which means
			# we after the ']' but not at the end of the string.
			raise TokenizeError(
				"Unable to process bracket atom as characters exist after the bracket close",
				smiles, start + match_start,
				current_state, [])

		m = lexer(term, match_start)
		if m is None:
			# None of the expected patterns matched
			raise TokenizeError(
				"Unable to process bracket atom after %s" % (current_state.replace("_", " "),),
				smiles, start + match_start,
				current_state, expected_states)

		# We have a match! Figure out which one.
		for new_state in expected_states:
			if m.group(new_state) is not None:
				current_state = new_state
				break
		else:
			# shouldn't be possible. All regexps are supposed to have a (?P<name>..) group
			raise AssertionError((term, start, current_state, expected_states))
		# Advance to the start of the next token
		match_start = m.end()

	# Reached the end of the string. Have we also reached the ']'?
	if current_state != "end_bracket":
		# No. This can only happen if the last term of the SMILES is
		# an incomplete bracket atom; missing the ']'.
		lexer, expected_states = _bracket_atom_tokenizer_states[current_state]
		raise TokenizeError(
			"Unable to process bracket atom as it is incomplete",
			smiles, start + match_start,
			current_state, expected_states)

	# If we get here then everything parsed correctly. But in that case the
	# regexp should have caught it, so getting here should be impossible.
	raise AssertionError(("Expected failure", smiles))


############ End of SMILES tokenizer. Let's process those tokens

## Figure out which atoms are involved in a branch.
#
# For something like A(BC)D I keep track of:
#  atom_token_index = 0 = the location of the atom ("A") where the branch comes from
#  open_token_index = 1 = the location of the '('
#  first_branch_token_index = 2 = the location of first atom in the branch ("B")
#  close_token_index = 4 = the location of the ')'

class Branch(object):
	def __init__(self, base_atom_token_index, open_token_index,
				 first_branch_atom_token_index, close_token_index):
		self.base_atom_token_index = base_atom_token_index
		self.open_token_index = open_token_index
		self.first_branch_atom_token_index = first_branch_atom_token_index
		self.close_token_index = close_token_index

def match_branches(tokens, smiles):
	# stack of (base_atom, open_branch token) pairs
	# the 'base_atom' is the atom just before the branch open token
	branch_stack = []
	base_atom = None
	for token in tokens:
		if token.typename == "atom":
			# Keep track of it in case we open a branch
			base_atom = token
		elif token.typename == "open_branch":
			# The previous atom we saw was the base atom
			branch_stack.append((base_atom, token))
		elif token.typename == "close_branch":
			if not branch_stack:
				raise ParseError("Close branch without an open branch",
								 smiles, token.start)
			# Get the base and open branch.
			# The base_atom can be a base_atom for the next closure
			base_atom, open_branch = branch_stack.pop()
			branch_atom_token_index = open_branch.token_index + 1
			if tokens[branch_atom_token_index].typename != "atom":
				assert tokens[branch_atom_token_index].typename == "bond", tokens[branch_atom_token_index]
				branch_atom_token_index += 1
				assert tokens[branch_atom_token_index].typename == "atom", tokens[branch_atom_token_index]

			yield Branch(base_atom.token_index, open_branch.token_index,
						 branch_atom_token_index, token.token_index)
	if branch_stack:
		raise ParseError("Open branch without a close branch",
						 smiles, branch_stack[0][1].start)

## Figure out where the closures are

def _parse_closure_value(term):
	# A closure can be in one of three forms:
	n = len(term)
	if n == 1:
		# 1) a single digit closure
		return int(term)
	elif n == 3:
		# 2) a '%' followed by two digits
		return int(term[1:])
	else:
		# 3) a '%(' followed by 1 or more digits followed by a ')'
		return int(term[2:-1])

class Closure(object):
	def __init__(self, closure_id,
				 first_atom, first_bond, first_closure,
				 second_atom, second_bond, second_closure,
				 bond_type):
		self.closure_id = closure_id
		self.first_atom = first_atom
		self.first_bond = first_bond
		self.first_closure = first_closure
		self.second_atom = second_atom
		self.second_bond = second_bond
		self.second_closure = second_closure
		self.bond_type = bond_type
	def __repr__(self):
		return "Closure(%d, %d, %r, %d, %d, %r, %d, %r)" % (
			self.closure_id,
			self.first_atom,
			self.first_bond,
			self.first_closure,
			self.second_atom,
			self.second_bond,
			self.second_closure,
			self.bond_type,
			)
	def __eq__(self, other):
		return (
			self.closure_id == other.closure_id and
			self.first_atom == other.first_atom and
			self.first_bond == other.first_bond and
			self.first_closure == other.first_closure and
			self.second_atom == other.second_atom and
			self.second_bond == other.second_bond and
			self.second_closure == other.second_closure and
			self.bond_type == other.bond_type)
	def __ne__(self, other):
		return not (self == other)


def find_closures(tokens, smiles):
	closure_table = {}
	atom = None
	for token in tokens:
		if token.typename == "atom":
			atom = token
			bond_token = None
		elif token.typename == "bond":
			bond_token = token
		elif token.typename == "closure":
			closure = _parse_closure_value(token.term)
			if closure in closure_table:
				prev_atom, prev_bond_token, prev_closure = closure_table[closure]
				if prev_atom.atom_index == atom.atom_index:
					raise ClosureError(
						"Cannot connect an atom to itself",
						smiles, token.start, prev_closure.start,
						"Trying to connect back to here.")

				if prev_bond_token is None:
					if bond_token is None:
						bond_type = None
					else:
						bond_type = bond_token.term
						# If the first closure has no direction
						# and the second one has a direction
						# then we need need to reverse it because
						# the closure.bond_type is done with respect
						# to the first atom.
						if bond_type == "/":
							bond_type = "\\"
						elif bond_type == "\\":
							bond_type = "/"
				elif bond_token is None:
					bond_type = prev_bond_token.term
				else:
					expected = prev_bond_token.term
					# If it is a directional bond, make sure it was flipped
					if expected == "/":
						expected = "\\"
					elif expected == "\\":
						expected = "/"
					if bond_token.term != expected:
						raise ClosureError(
							"Mismatch in closure bond type",
							smiles, bond_token.start, prev_bond_token.start,
							"Should match closure bond here",
							advice = "Expecting '%s' or unspecified bond type." % (expected,))
					bond_type = prev_bond_token.term

				first_bond = (None if prev_bond_token is None else prev_bond_token.token_index)
				second_bond = (None if bond_token is None else bond_token.token_index)

				yield Closure(closure, prev_atom.token_index, first_bond, prev_closure.token_index,
							  atom.token_index, second_bond, token.token_index, bond_type)
				del closure_table[closure]
			else:
				closure_table[closure]  = (atom, bond_token, token)
			bond_token = None
		else:
			bond_token = None

	if closure_table:
		start, closure = min( (v[2].start, k) for (k, v) in closure_table.items())
		raise ParseError(
			"Unclosed closure (%d)" % (closure,), smiles, start)

class GraphAtom(object):
	def __init__(self, index, token):
		self.index = index
		self.token = token
		self.bond_indices = []
		self.neighbor_indices = []

	@property
	def symbol(self):
		return self.token.symbol
	@property
	def atomic_number(self):
		return self.token.atomic_number
	@property
	def is_aromatic(self):
		return self.token.is_aromatic
	@property
	def is_heavy(self):
		return self.token.is_heavy

class GraphBond(object):
	def __init__(self, index, token):
		self.index = index
		self.token = token
		self.atom_indices = []
		if token is None:
			self.bond_type = None
		else:
			self.bond_type = token.term
	def get_bond_type(self, from_atom=None, if_missing=None):
		bond_type = self.bond_type
		if (from_atom is not None
			and from_atom == self.atom_indices[1]):
			# Need to reverse the direction
			if bond_type == "/":
				return "\\"
			elif bond_type == "\\":
				return "/"
		if bond_type is None:
			return if_missing
		return bond_type

class GraphClosureBond(GraphBond):
	def __init__(self, index, token, first_closure_token):
		super(GraphClosureBond, self).__init__(index, token)
		self.first_bond_token = token
		self.first_closure_token = first_closure_token
		self.second_bond_token = None
		self.second_closure_token = None
	def finish(self, second_bond_token, second_closure_token):
		assert self.second_bond_token is None
		self.second_bond_token = second_bond_token
		self.second_closure_token = second_closure_token
		# Figure out the bond type. The first bond token has priority.
		# Any conflict should have been detected by find_closures().
		if self.first_bond_token is None:
			if second_bond_token is not None:
				bond_type = second_bond_token.bond_type
				if bond_type == "/":
					bond_type = "\\"
				elif bond_type == "\\":
					bond_type = "/"
				self.bond_type = bond_type

class Graph(object):
	def __init__(self, atoms, bonds):
		self.atoms = atoms
		self.bonds = bonds
	def dump(self):
		print("%d atom %d bonds" % (len(self.atoms), len(self.bonds)))
		print("Atoms:")
		for atom in self.atoms:
			to_xatoms = ["%d->%d(%s%s)" % (
				bond_index, atom_index,
				self.bonds[bond_index].get_bond_type(atom.index, "~"), self.atoms[atom_index].token.term)
							   for bond_index, atom_index in zip(atom.bond_indices, atom.neighbor_indices)]
			if not to_xatoms:
				to_xatoms = "(disconnected)"
			else:
				to_xatoms = ", ".join(to_xatoms)
			print(" %d %s bonds: %s" % (atom.index, atom.token.term, to_xatoms))
		print("Bonds:")
		for bond in self.bonds:
			atom1_index, atom2_index = bond.atom_indices
			atoms_str = "[%d,%d] (%s%s%s)" % (
				atom1_index, atom2_index,
				self.atoms[atom1_index].token.term,
				bond.bond_type or "~",
				self.atoms[atom2_index].token.term)

			if isinstance(bond, GraphClosureBond):
				print(" %d %s atoms: %s (closure %d)" % (
					bond.index, bond.bond_type or "~", atoms_str,
					_parse_closure_value(bond.first_closure_token.term)))
			else:
				print(" %d %s atoms: %s" % (bond.index, bond.bond_type or "~", atoms_str))
		print("End of graph.")

# I wanted to include "fragments" in the default output.
# I don't want to require RDKit by default, because I want
# this code to work even if RDKit isn't available.
# Which means I can't use Chem.GetMolFrags().
# So I'll make my own. Which means I need a graph-like object.
# Here it is.
def make_graph(tokens):
	closure_table = {}
	prev_graph_atom = None
	prev_bond_token = None
	branch_stack = []

	graph_atoms = []
	graph_bonds = []

	for token in tokens:
		if token.typename == "atom":
			graph_atom = GraphAtom(len(graph_atoms), token)
			graph_atoms.append(graph_atom)

			if prev_graph_atom is not None:
				# Then there is a bond, either implicit (prev_bond_token is
				# None) or explicit
				new_bond_index = len(graph_bonds)
				graph_bond = GraphBond(new_bond_index, prev_bond_token)
				graph_bonds.append(graph_bond)

				prev_graph_atom.bond_indices.append(new_bond_index)
				graph_atom.bond_indices.append(new_bond_index)
				graph_bond.atom_indices.append(prev_graph_atom.index)
				graph_bond.atom_indices.append(graph_atom.index)

				prev_graph_atom.neighbor_indices.append(graph_atom.index)
				graph_atom.neighbor_indices.append(prev_graph_atom.index)

			prev_graph_atom = graph_atom
			prev_bond_token = None

		elif token.typename == "bond":
			prev_bond_token = token

		elif token.typename == "dot":
			prev_graph_atom = None
			prev_bond_token = None

		elif token.typename == "open_branch":
			assert prev_graph_atom is not None, "branch without previous atom?"
			assert prev_bond_token is None, "how did a bond get before the '('?"
			branch_stack.append(prev_graph_atom)

		elif token.typename == "close_branch":
			assert prev_bond_token is None, "how did a bond get before the ')'?"
			if branch_stack:
				prev_graph_atom = branch_stack.pop()
			# Ignore an empty stack. It should be caught in match_branches()

		elif token.typename == "closure":
			assert prev_graph_atom is not None, "closure without previous atom?"
			closure = _parse_closure_value(token.term)
			if closure in closure_table:
				closure_graph_atom, closure_graph_bond, closure_token = closure_table.pop(closure)
				closure_graph_bond.finish(prev_bond_token, token)
				# Fill in the other 1/2 of the bond
				first_atom_index, = closure_graph_bond.atom_indices
				if first_atom_index in prev_graph_atom.neighbor_indices:
					smiles = "".join(token.term for token in tokens)
					raise ClosureError(
						"Closure may not connect two atoms which are already connected",
						smiles, token.start, closure_token.start,
						"Start of the closure")

				prev_graph_atom.bond_indices.append(closure_graph_bond.index)
				prev_graph_atom.neighbor_indices.append(first_atom_index)
				closure_graph_bond.atom_indices.append(prev_graph_atom.index)
				graph_atoms[first_atom_index].neighbor_indices.append(prev_graph_atom.index)
			else:
				# Make a 1/2 bond
				new_bond_index = len(graph_bonds)
				graph_bond = GraphClosureBond(new_bond_index, prev_bond_token, token)
				graph_bonds.append(graph_bond)
				graph_atom.bond_indices.append(new_bond_index)
				graph_bond.atom_indices.append(graph_atom.index)
				closure_table[closure] = graph_atom, graph_bond, token
			prev_bond_token = None
		else:
			raise AssertionError(("Unknown token", token))

	# Fill in the missing closures with unspecified atoms
	# (Shouldn't happen if find_closures() did its job.)
	for closure, (graph_atom, graph_bond) in closure_table.items():
		new_atom_index = len(graph_atoms)
		new_graph_atom = GraphAtom(new_atom_index, Token(None, "atom", "*", None, None))
		graph_atoms.append(new_graph_atom)
		new_graph_atom.bond_indices.append(graph_bond.index)
		graph_bond.atom_indices.append(new_atom_index)

	## # fill in the atom neighbors
	## for graph_atom in graph_atoms:
	##     neighbor_indices = graph_atom.neighbor_indices
	##     for bond_index in graph_atom.bond_indices:
	##         graph_bond = graph_bonds[bond_index]
	##         assert len(graph_bond.atom_indices) == 2
	##         if graph_bond.atom_indices[0] == graph_atom.index:
	##             index_to_add = graph_bond.atom_indices[1]
	##         else:
	##             index_to_add = graph_bond.atom_indices[0]
	##         if index_to_add in neighbor_indices:
	##             # There's already
	##             raise AssertionError("spam")
	##         neighbor_indices.append(index_to_add)

	return Graph(graph_atoms, graph_bonds)

def get_graph_fragments(graph):
	graph_atoms = graph.atoms
	graph_bonds = graph.bonds

	seen = set()
	fragments = []

	for graph_atom in graph_atoms:
		seed = graph_atom.index
		if seed in seen:
			continue
		fragment = [seed]
		stack = [seed]
		seen.add(seed)

		while stack:
			atom_idx = stack.pop()
			for neighbor_idx in graph_atoms[atom_idx].neighbor_indices:
				if neighbor_idx in seen:
					continue
				fragment.append(neighbor_idx)
				stack.append(neighbor_idx)
				seen.add(neighbor_idx)
		fragments.append(tuple(fragment))
	return tuple(fragments)

def verify_get_mol_frags():
	from rdkit import Chem
	filename = "/Users/dalke/databases/chembl_23.rdkit.smi"
	with open(filename) as infile:
		for lineno, line in enumerate(infile):
			if lineno % 1000 == 0:
				sys.stderr.write("Processed %d lines.\n" % (lineno,))
			try:
				smiles, id = line.split()
			except ValueError:
				print
			try:
				tokens = list(tokenize_smiles(smiles))
			except ParseError as err:
				msg = "Cannot parse SMILES\n"
				msg += err.get_failure_highlight(prefix="  ")
				msg += err.get_advice()
				print(msg)
				continue

			graph = make_graph(tokens)
			my_mol_frags = get_graph_fragments(graph)
			mol = Chem.MolFromSmiles(smiles)
			if mol is None:
				continue
			rd_mol_frags = Chem.GetMolFrags(mol)
			my_mol_frags = list(map(sorted, my_mol_frags))
			rd_mol_frags = list(map(sorted, rd_mol_frags))
			if my_mol_frags != rd_mol_frags:
				raise AssertionError( (smiles, my_mol_frags, rd_mol_frags) )
#verify_get_graph_fragments()

class MolGraph(object):
	def __init__(self, mol, atoms, bonds):
		self.mol = mol
		self.atoms = atoms
		self.bonds = bonds

class MolGraphAtom(object):
	def __init__(self, mol_atom, atom_symbol, index, token, bond_indices, neighbor_indices, bond_types):
		self.mol_atom = mol_atom
		self.atom_symbol = atom_symbol
		self.index = index
		self.token = token
		self.bond_indices = bond_indices
		self.neighbor_indices = neighbor_indices
		self.bond_types = bond_types
	def get_outgoing(self):
		return zip(self.bond_indices, self.bond_types, self.neighbor_indices)

class MolGraphBond(object):
	def __init__(self, mol_bond, index, token, atom_indices, bond_type):
		self.mol_bond = mol_bond
		self.index = index
		self.token = token
		self.atom_indices = atom_indices
		self.bond_type = bond_type

	def get_bond_type(self, from_atom=None, if_missing=None):
		bond_type = self.bond_type
		if (from_atom is not None
			and from_atom == self.atom_indices[1]):
			# Need to reverse the direction
			if bond_type == "/":
				return "\\"
			elif bond_type == "\\":
				return "/"
		if bond_type is None:
			return if_missing
		return bond_type

def make_mol_graph(graph, mol, mol_atoms, mol_atom_symbols, bond_symbol_table):
	assert len(graph.atoms) == len(mol_atoms)
	assert bond_symbol_table is not None

	mol_graph_atoms = []
	for atom, graph_atom, atom_symbol in zip(mol_atoms, graph.atoms, mol_atom_symbols):
		mol_graph_atoms.append(
			MolGraphAtom(
				atom, atom_symbol, graph_atom.index, graph_atom.token,
				graph_atom.bond_indices[:], graph_atom.neighbor_indices[:],
				[])
			)
	mol_graph_bonds = []
	for graph_bond in graph.bonds:
		left_atom_idx, right_atom_idx = graph_bond.atom_indices
		key = left_atom_idx, right_atom_idx
		if key in bond_symbol_table:
			bond, bond_type = bond_symbol_table[key]
		else:
			key = right_atom_idx, left_atom_idx
			if key in bond_symbol_table:
				# XXX Need to reverse "/" and "\\"?
				bond, bond_type = bond_symbol_table[key]
			else:
				raise AssertionError(("Failed to find", key))

		mol_graph_bond = MolGraphBond(bond, graph_bond.index, graph_bond.token,
									  graph_bond.atom_indices, bond_type)
		mol_graph_bonds.append(mol_graph_bond)

	for mol_graph_atom in mol_graph_atoms:
		bond_types = mol_graph_atom.bond_types
		for bond_idx, neighbor_idx in zip(mol_graph_atom.bond_indices,
										  mol_graph_atom.neighbor_indices):
			mol_graph_bond = mol_graph_bonds[bond_idx]
			# XXX swap direction?
			bond_types.append(mol_graph_bond.bond_type)

	return MolGraph(mol, mol_graph_atoms, mol_graph_bonds)


############## End of SMILES syntax processing

############## Property dependency handler

# On-demand calculation (and caching) of molecular properties.

# This approach is based on some ideas I first started doing around
# 2000 and have used in a couple of projects. This is more of a sketch
# of some ideas I have in how to handle 'namespaces'.

# This is not meant to end-user library code. It's more of a
# playground to experiment with this approach.


# A function which takes descriptors and returns one descriptor.
class PropertyRule(object):
	def __init__(self, func, input_names, output_name):
		self.func = func
		self.input_names = input_names
		self.output_name = output_name
		self.output_names = [output_name]

	def __repr__(self):
		return "PropertyRule(%r, %r, %r)" % (self.func, self.input_names, self.output_name)

	def compute(self, propname, properties):
		# Pull off the needed properties, call the function, and save/return the result.
		inputs = [properties[name] for name in self.input_names]
		result = self.func(*inputs)
		properties[self.output_name] = result

	def get_name(self):
		return self.func.__name__

# A function which takes descriptors and returns a (fixed) list of descriptors
class MultiPropertyRule(object):
	def __init__(self, func, input_names, output_names):
		self.func = func
		self.input_names = input_names
		self.output_names = output_names

	def __repr__(self):
		return "MultiPropertyRule(%r, %r, %r)" % (self.func, self.input_names, self.output_names)

	def compute(self, propname, properties):
		# Pull off the needed properties, call the function, and save/return the result.
		inputs = [properties[name] for name in self.input_names]
		results = self.func(*inputs)
		if not isinstance(results, tuple):
			raise TypeError("%r must return a tuple of %r. Got: %r" % (
				self.func, tuple(self.output_names), results))
		if len(results) != len(self.output_names):
			raise TypeError("%r returned %d values, expected %d. Got: %r" % (
				self.func, len(results), len(self.output_names), results))
			raise AssertionError((results, self.output_names))
		for name, value in zip(self.output_names, results):
			properties[name] = value

	def get_name(self):
		return self.func.__name__

# The "dyanmic" variants are passed in the properties dictionary so
# they can look up a value on-demand, rather than pass in the
# properties. This is more general-purpose, but experience shows that
# it's better to be explicit about which properties are
# needed/returned.

class DynamicPropertyRule(object):
	def __init__(self, func, input_names, output_name):
		self.func = func
		self.input_names = input_names
		self.output_name = output_name
		self.output_names = [output_name]

	def __repr__(self):
		return "DynamicPropertyRule(%r, %r, %r)" % (self.func, self.input_names, self.output_name)

	def compute(self, propname, properties):
		inputs = [propname, properties] + [properties[name] for name in self.input_names]
		result = self.func(*inputs)
		properties[self.output_name] = result

	def get_name(self):
		return self.func.__name__

class DynamicMultiPropertyRule(object):
	def __init__(self, func, input_names, output_names):
		self.func = func
		self.input_names = input_names
		self.output_names = output_names

	def __repr__(self):
		return "DynamicPropertyRule(%r, %r, %r)" % (self.func, self.input_names, self.output_names)

	def compute(self, propname, properties):
		inputs = [propname, properties] + [properties[name] for name in self.input_names]
		results = self.func(*inputs)
		if not isinstance(results, tuple):
			raise TypeError("%r must return a tuple of %r. Got: %r" % (
				self.func, tuple(self.output_names)), results)
		if len(results) != len(self.output_names):
			raise TypeError("%r returned %d values, expected %d. Got: %r" % (
				self.func, len(results), len(self.output_names), results))
			raise AssertionError((results, self.output_names))
		for name, value in zip(self.output_names, results):
			properties[name] = value

	def get_name(self):
		return self.func.__name__

# A rule which forwards to a set of rules, in a different namespace.
class NamespaceRule(object):
	def __init__(self, namespace_name, ruleset, forward_defaults=None, mapped_properties=None):
		# Copy defaults from the parent namespace
		if forward_defaults is None:
			forward_defaults = {}
		# Property lookups from the parent are mapped to the new namespace
		if mapped_properties is None:
			mapped_properties = {}

		self.namespace_name = namespace_name
		self.ruleset = ruleset
		self.forward_defaults = forward_defaults
		self.mapped_properties = mapped_properties

		self.output_names = list(sorted(self.mapped_properties))

	def __repr__(self):
		return "NamespaceRule(%r, <%d subrules>, <%d forwards>, <%d mapped properties>)" % (
			self.namespace_name, len(self.ruleset), len(self.forward_defaults), len(self.mapped_properties))

	def compute(self, propname, properties):
		sub_propname = self.mapped_properties[propname]
		value = properties.property_namespaces[self.namespace_name][sub_propname]
		properties[propname] = value

	def get_properties(self, parent_values, namespace_values):
		# Requested from the parent
		child_initial_values = {}
		for parent_name, child_name in self.forward_defaults.items():
			if parent_name in parent_values:
				child_initial_values[child_name] = parent_values[parent_name]

		# Overridden by the user
		child_initial_values.update(namespace_values)

		return self.ruleset.get_properties(child_initial_values)

	def dump(self, indent=0, parent_name=None):
		prefix = " " * indent
		if parent_name is None:
			name = self.namespace_name
		else:
			name = "%s.%s" % (parent_name, self.namespace_name)
		print(prefix + "Namespace %r" % (name,))
		print(prefix + " Defaults:")
		for k, v in sorted(self.forward_defaults.items()):
			print(prefix + "  %r -> %r" % (k, v))
		print(prefix + " Mapped properties:")
		for k, v in sorted(self.mapped_properties.items()):
			print(prefix + "  %r -> %r" % (k, v))
		self.ruleset.dump(indent+1, name)


class Ruleset(object):
	def __init__(self, rules_by_property=None, default_property_values=None, property_namespaces=None):
		if rules_by_property is None:
			rules_by_property = {}
		if default_property_values is None:
			default_property_values = {}
		if property_namespaces is None:
			property_namespaces = {}

		self.rules_by_property = rules_by_property
		self.default_property_values = default_property_values
		self.property_namespaces = property_namespaces

	def copy(self):
		return Ruleset(self.rules_by_property.copy(), self.default_property_values.copy(),
					   dict((k, v.copy()) for (k, v) in self.property_namespaces.items()))


	def add_property_rule(self, property_rule):
		# What about "." names?
		for name in property_rule.output_names:
			#print("Added", name, "to", property_rule)
			self.rules_by_property[name] = property_rule

	def __iter__(self):
		return iter(self.rules_by_property)

	def __len__(self):
		return len(self.rules_by_property)

	def __getitem__(self, propname):
		return self.rules_by_property[propname]

	def __contains__(self, propname):
		return propname in self.rules_by_property

	def get_properties(self, initial_values=None):
		my_properties = self.default_property_values.copy()
		# Create a namespace for each of the known subproperties
		namespace_properties = dict((name, {}) for name in self.property_namespaces)
		if initial_values is not None:
			for propname, value in initial_values.items():
				left, mid, right = propname.partition(".")
				# If it's a known dot name, put the value in the namespace
				if mid:
					if left in namespace_properties:
						namespace_properties[left][right] = value
				else:
					# Not a dot name
					my_properties[propname] = value

		# Create the childern properties and initialize my own properties
		namespaces = {}
		for name, namespace in self.property_namespaces.items():
			child_properties = namespace_properties[name]
			namespaces[name] = namespace.get_properties(my_properties, child_properties)

		return Properties(self, my_properties, namespaces)

	def __add__(self, other):
		new_ruleset = self.copy()
		new_ruleset.update(other)
		return new_ruleset

	def update(self, ruleset):
		self.rules_by_property.update(ruleset.rules_by_property)
		self.default_property_values.update(ruleset.default_property_values)
		self.property_namespaces.update(ruleset.property_namespaces)

	def add_namespace(self, name, ruleset, forward_defaults=None, mapped_properties=None):
		if forward_defaults is None:
			forward_defaults = {}
		if mapped_properties is None:
			mapped_properties = {}
		if name in self.property_namespaces:
			raise ValueError("namespace %r already defined" % (name,))
		self.property_namespaces[name] = namespace_rule = NamespaceRule(
			name, ruleset, forward_defaults=forward_defaults, mapped_properties=mapped_properties)
		self.add_property_rule(namespace_rule)
		#return namespace


	def get_rule_decorator(self):
		def add_rule(*output_names, **kwargs):
			if "dynamic" in kwargs:
				dynamic = kwargs.pop("dynamic")
			else:
				dynamic = False
			if kwargs:
				raise ValueError("Unknown kwargs: %r" % (kwargs,))
			# Python 3
			def add_rule_func(func):
				argspec = inspect.getargspec(func)
				input_names = argspec.args
				for name in input_names:
					if not isinstance(name, str):
						raise AssertionError(("args must only contain strings", argspec))

				if not output_names:
					func_name = getattr(func, "__name__", "")
					if func_name[:4] == "get_" and len(func_name) > 5:
						# Can't set 'output_names' because that's defined in an outer scope.
						# Easiest is to make a new local variable.
						local_output_names = [func_name[4:]]
					else:
						raise ValueError("must specify at least one output name when the "
										 "function name is not in the format 'get_{name}'")
				else:
					local_output_names = output_names


				if dynamic:
					if len(input_names) < 2:
						raise ValueError("dynamic ruleset must accept at least two parameters; 'propname' and 'properties'")
					# Check that the first two args have no defaults
					if len(argspec.defaults) > len(input_names) - 2:
						raise ValueError("first two parameters must not have default values")

					if len(local_output_names) == 1:
						property_rule = DynamicPropertyRule(
							func, input_names[2:], local_output_names[0])
					else:
						property_rule = DynamicMultiPropertyRule(
							func, input_names[2:], local_output_names)
				else:
					if len(local_output_names) == 1:
						property_rule = PropertyRule(
							func, input_names, local_output_names[0])
					else:
						property_rule = MultiPropertyRule(
							func, input_names, local_output_names)

				defaults = argspec.defaults
				if defaults:
					#print("defaults", input_names, defaults)
					for i in range(-len(defaults), 0):
						#print("set default", input_names[i], defaults[i])
						self.default_property_values[input_names[i]] = defaults[i]
				self.add_property_rule(property_rule)
				return func
			return add_rule_func
		return add_rule

	def dump(self, indent=0, ruleset_name=None):
		prefix = " " * indent
		if ruleset_name is None:
			print(prefix + "Ruleset:")
		else:
			print(prefix + "Ruleset: %r" % (ruleset_name,))
		print(prefix + " Defaults:")
		for name, value in sorted(self.default_property_values.items()):
			print(prefix + "  %s: %r" % (name, value))
		print(prefix + " Properties:")
		for name, value in sorted(self.rules_by_property.items()):
			print(prefix + "  %s: %r" % (name, value))
		print(prefix + " Namespaces:")
		for name, value in sorted(self.property_namespaces.items()):
			if ruleset_name is None:
				value.dump(indent+2, name)
			else:
				value.dump(indent+2, ruleset_name + "." + name)


class Properties(object):
	def __init__(self, ruleset, initial_values, property_namespaces):
		self.ruleset = ruleset
		self.cache = initial_values
		self.property_namespaces = property_namespaces

	def __getitem__(self, propname):
		left, mid, right = propname.partition(".")
		if mid:
			return self.property_namespaces[left][right]

		# Check if I know about it
		try:
			return self.cache[propname]
		except KeyError:
			pass

		rule = self.ruleset[propname]
		recursion_check = RecursionCheck(self, propname, self.property_namespaces)
		rule.compute(propname, recursion_check)

		return self.cache[propname]

	def is_in_cache(self, propname):
		left, mid, right = propname.partition(".")
		if mid:
			return self.property_namespaces[left].is_in_cache(right)
		return propname in self.cache

	def get_cached_value_or_rule(self, propname):
		#print("Looking up", propname)
		left, mid, right = propname.partition(".")
		if mid:
			return self.property_namespaces[left].get_cached_value_or_rule(right)
		try:
			return self.cache[propname], None
		except KeyError:
			pass
		return None, self.ruleset[propname]

	def __setitem__(self, propname, value):
		assert propname not in self.cache, value
		self.cache[propname] = value

class RecursionCheck(object):
	def __init__(self, properties, start_propname, property_namespaces):
		self.properties = properties
		self.seen = set()
		self.stack = [start_propname]
		self.property_namespaces = property_namespaces

	def __getitem__(self, propname):
		#print("Looking for %r (in %s) Stack: %r" % (propname, self.properties.ruleset, self.stack))
		try:
			cached_value, rule = self.properties.get_cached_value_or_rule(propname)
		except Exception:
			sys.stderr.write("property stack for %r: %r\n" % (propname, " -> ".join(self.stack)))
			raise
		if rule is None:
			return cached_value
		if propname in self.seen:
			dependencies = []
			for name in self.stack:
				v, r = self.properties.get_cached_value_or_rule(name)
				if r is None:
					dependencies.append("%r (cached)" % (name,))
				else:
					dependencies.append("%r (%r)" % (name, r.get_name()))
			raise ValueError("Dependency loop detected trying to compute property %r (%s)"
							  % (propname, " -> ".join(dependencies)))

		self.seen.add(propname)
		self.stack.append(propname)
		try:
			rule.compute(propname, self)
		finally:
			self.stack.pop()

		cached_value, rule = self.properties.get_cached_value_or_rule(propname)
		#print("Got", propname, cached_value, rule)
		if rule is not None:
			raise AssertionError("should have found %r" % (propname,))
		return cached_value

	def __setitem__(self, propname, value):
		self.properties[propname] = value

############  End of property dependency handler

############ Define some rulesets

## Base ruleset. These depend on a "smiles" property.

base_ruleset = Ruleset()
add_base_rule = base_ruleset.get_rule_decorator()

@add_base_rule()
def get_tokens(smiles):
	return list(tokenize_smiles(smiles))

@add_base_rule()
def get_atom_tokens(tokens):
	return [token for token in tokens if token.typename == "atom"]

@add_base_rule()
def get_closures(smiles, tokens):
	return list(find_closures(tokens, smiles))

@add_base_rule()
def get_branches(smiles, tokens):
	return list(match_branches(tokens, smiles))

@add_base_rule("graph")
def get_graph(tokens):
	return make_graph(tokens)

@add_base_rule("fragments")
def get_mol_fragments(graph):
	return get_graph_fragments(graph)

@add_base_rule()
def get_check_smiles_syntax(tokens, branches, closures, graph):
	# If all of the properties can be computed then the syntax is correct.
	# If one of them couldn't be computed, then the dependencies
	# will raise a ParseError
	return True

#### These are for input processing

input_ruleset = base_ruleset.copy()

add_input_rule = input_ruleset.get_rule_decorator()

@add_input_rule()
def get_atom_indices(atom_tokens):
	return list(range(len(atom_tokens)))

@add_input_rule(dynamic=True)
def get_new_isotope_values(propname, properties, set_isotope="none"):
	assert set_isotope is not None
	return get_atom_values(properties, set_isotope, prefix="", parameter_name="set_isotope")

@add_input_rule(dynamic=True)
def get_new_atom_class_values(propname, properties, set_atom_class="none"):
	return get_atom_values(properties, set_atom_class, prefix=":", parameter_name="set_atom_class")

# Define "index", "index+1", "index+10", "index+100",
# and similar variants for for "eleno", and "symclass".

_source_lookup = {
	"index": (True, "atom_indices"),
	"eleno": (True, "element_numbers"),
	"symclass": (True, "symmetry_classes"),
}
_sources = {}
def _init_sources():
	for name, (allow_offsets, propname) in _source_lookup.items():
		_sources[name] = (propname, 0)
		if allow_offsets:
			for offset in (1, 10, 100):
				_sources["%s+%d" % (name, offset)] = (propname, offset)
_init_sources()

def get_atom_values(properties, source, prefix, parameter_name):
	if source == "none":
		return None

	if source == "remove":
		return [prefix for atom in properties["atom_tokens"]]

	if source not in _sources:
		raise ValueError(
			"Unknown %r value %r. Must be one of %r"
			% (parameter_name, source, sorted(_sources)))

	propname, offset = _sources[source]
	values = properties[propname]
	if values is None:
		return None
	return [(prefix + str(value+offset)) for value in values]

## input_ruleset.add_defaults(new_isotope_values="none",
##                            new_atom_class_values="none",

@add_input_rule("modified_smiles", "smiles_is_modified", dynamic=True)
def get_modified_tokens(
		propname, properties,
		smiles,
		tokens,
		new_isotope_values,
		new_atom_class_values,
		use_brackets=False,
		):
	if new_isotope_values is not None or new_atom_class_values is not None:
		use_brackets = True
		will_modify = True
	else:
		will_modify = False

	if not use_brackets:
		return (smiles, False)

	if not will_modify:
		need_hcounts = False
		for token in tokens:
			if token.typename == "atom":
				if token.subtypename == "organic_atom":
					need_hcounts = True
					break
		if not need_hcounts:
			return (smiles, False)

	hcounts = properties["hcounts"]

	terms = []
	for token in tokens:
		if token.typename == "atom":
			if token.subtypename == "organic_atom":
				term = token.to_bracket_format(hcounts[token.atom_index])
				new_token = tokenize_bracket_atom(term)
			else:
				new_token = token
			term = '['
			if new_isotope_values is not None:
				term += str(new_isotope_values[token.atom_index])
			else:
				term += new_token.isotope
			term += new_token.symbol + new_token.chiral + new_token.hcount + new_token.charge
			if new_atom_class_values is not None:
				term += str(new_atom_class_values[token.atom_index])
			else:
				term += new_token.atom_class
			term += "]"
		else:
			term = token.term

		terms.append(term)

	modified_smiles = "".join(terms)
	return modified_smiles, True


#### These depend on the RDKit toolkit

rdkit_ruleset = Ruleset()
add_rdkit_rule = rdkit_ruleset.get_rule_decorator()

@add_rdkit_rule("mol", "mol_errmsg")
def get_mol(smiles, sanitize=True):
	from rdkit import Chem
	#print("SMILES", repr(smiles))
	mol_errmsg = None
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		mol_errmsg = "RDKit cannot parse the SMILES"
	elif sanitize:
		Chem.SanitizeMol(mol)
	else:
		# Don't modify bond orders or charges, which can (e.g.)
		# happen around nitro groups.
		Chem.SanitizeMol(mol,Chem.SANITIZE_ALL^Chem.SANITIZE_CLEANUP^Chem.SANITIZE_PROPERTIES)

	return mol, mol_errmsg

@add_rdkit_rule()
def get_mol_atoms(mol):
	return list(mol.GetAtoms())

@add_rdkit_rule()
def get_mol_bonds(mol):
	return list(mol.GetBonds())

@add_rdkit_rule()
def get_element_numbers(mol_atoms):
	return [mol_atom.GetAtomicNum() for mol_atom in mol_atoms]

@add_rdkit_rule()
def get_symmetry_classes(mol):
	from rdkit import Chem
	return Chem.CanonicalRankAtoms(mol, breakTies=False)

@add_rdkit_rule()
def get_hcounts(mol_atoms):
	return [atom.GetNumImplicitHs()+atom.GetNumExplicitHs() for atom in mol_atoms]

@add_rdkit_rule()
def get__rdkit_atom_rings(mol):
	return [set(atom_indices) for atom_indices in mol.GetRingInfo().AtomRings()]

@add_rdkit_rule()
def get_mol_atom_symbols(mol_atoms):
	symbols = []
	for atom in mol_atoms:
		symbol = atom.GetSymbol()
		if atom.GetIsAromatic():
			symbol = symbol.lower()
		symbols.append(symbol)
	return symbols

@add_rdkit_rule()
def get_mol_bond_symbol_table(mol_bonds):
	from rdkit import Chem
	bond_type_labels = {
		Chem.BondType.SINGLE: "-",
		Chem.BondType.DOUBLE: "=",
		Chem.BondType.AROMATIC: ":",
		Chem.BondType.TRIPLE: "#",
		}

	symbol_table = {}
	for bond in mol_bonds:
		bond_type = bond_type_labels[bond.GetBondType()]
		left_atom_idx, right_atom_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
		symbol_table[left_atom_idx, right_atom_idx] = (bond, bond_type)
	return symbol_table

@add_rdkit_rule()
def get_mol_graph(graph, mol, mol_atoms, mol_atom_symbols, mol_bond_symbol_table):
	return make_mol_graph(
		graph, mol, mol_atoms, mol_atom_symbols, mol_bond_symbol_table)


def has_rdkit():
	try:
		from rdkit import Chem
	except ImportError as err:
		return False
	return True

class rdkit_toolkit_api:
	toolkit_name = "rdkit"
	@staticmethod
	def get_matcher(smarts):
		from rdkit import Chem
		pattern = Chem.MolFromSmarts(smarts)
		return pattern # Need to use a portable API XXX

	@staticmethod
	def find_smallest_ring(properties, from_atom_index, to_atom_index):

		atom_rings = properties["_rdkit_atom_rings"]
		for atom_indices in atom_rings:
			if from_atom_index in atom_indices and to_atom_index in atom_indices:
				return atom_indices
		return {from_atom_index, to_atom_index}

	@staticmethod
	def get_substructure_matches(mol, pattern, uniquify=True, useChirality=False, maxMatches=1000):
		return mol.GetSubstructMatches(pattern, uniquify=uniquify, useChirality=useChirality, maxMatches=maxMatches)

####
#### These depend on the OEChem toolkit

openeye_ruleset = Ruleset()
add_openeye_rule = openeye_ruleset.get_rule_decorator()

@add_openeye_rule("mol", "mol_errmsg")
def get_mol(smiles):
	from openeye.oechem import OEGraphMol, OESmilesToMol, OEPerceiveSymmetry
	mol = OEGraphMol()
	mol_errmsg = None
	if not OESmilesToMol(mol, smiles):
		mol_errmsg = "OEChem cannot parse the SMILES"
		mol = None
	else:
		OEPerceiveSymmetry(mol)
	return mol, mol_errmsg

@add_openeye_rule()
def get_mol_atoms(mol):
	return list(mol.GetAtoms())

@add_openeye_rule()
def get_mol_bonds(mol):
	return list(mol.GetBonds())

@add_openeye_rule()
def get_element_numbers(mol_atoms):
	return [mol_atom.GetAtomicNum() for mol_atom in mol_atoms]

@add_openeye_rule()
def get_symmetry_classes(mol_atoms):
	return [a.GetSymmetryClass() for a in mol_atoms]

@add_openeye_rule()
def get_hcounts(mol_atoms):
	return [atom.GetImplicitHCount() for atom in mol_atoms]

@add_openeye_rule()
def get_mol_atom_symbols(mol_atoms):
	from openeye.oechem import OEGetAtomicSymbol
	symbols = []
	for atom in mol_atoms:
		eleno = atom.GetAtomicNum()
		if eleno == 0:
			symbol = "*"
		else:
			symbol = OEGetAtomicSymbol(eleno)
			if atom.IsAromatic():
				symbol = symbol.lower()
		symbols.append(symbol)
	return symbols

@add_openeye_rule()
def get_mol_graph(graph, mol, mol_atoms, mol_atom_symbols, mol_bond_symbol_table):
	return make_mol_graph(
		graph, mol, mol_atoms, mol_atom_symbols, mol_bond_symbol_table)

@add_openeye_rule()
def get_mol_bond_symbol_table(mol_bonds):
	_bond_symbols = {
		1: "-",
		2: "=",
		3: "#",
		4: "$",
		}
	symbol_table = {}
	for bond in mol_bonds:
		if bond.IsAromatic():
			bond_type = ":"
		else:
			bond_type = _bond_symbols[bond.GetOrder()]
		left_atom_idx, right_atom_idx = bond.GetBgnIdx(), bond.GetEndIdx()
		symbol_table[left_atom_idx, right_atom_idx] = (bond, bond_type)
	return symbol_table


def has_openeye():
	try:
		from openeye import oechem
	except ImportError as err:
		return False
	if oechem.OEChemIsLicensed():
		return True
	return False

class openeye_toolkit_api:
	toolkit_name = "openeye"
	@staticmethod
	def get_matcher(smarts):
		from openeye.oechem import OESubSearch
		subsearch = OESubSearch()
		if not subsearch.Init(smarts):
			return None
		return subsearch

	@staticmethod
	def find_smallest_ring(properties, from_atom_index, to_atom_index):
		from openeye.oechem import OEBondGetSmallestRingSize, OEGraphMol, OESubSearch
		mol = properties["mol"]
		mol_atoms = properties["mol_atoms"]
		a1 = mol_atoms[from_atom_index]
		a2 = mol_atoms[to_atom_index]
		bond = mol.GetBond(a1, a2)
		if bond is None:
			raise AssertionError("how? %r %d-%d" % (properties["smiles"], from_atom_index, to_atom_index))
			return {from_atom_index, to_atom_index}
		size = OEBondGetSmallestRingSize(bond)
		if not size:
			return {from_atom_index, to_atom_index}

		copy_mol = OEGraphMol(mol)
		for atom in copy_mol.GetAtoms():
			idx = atom.GetIdx()
			if idx == from_atom_index or idx == to_atom_index:
				atom.SetIsotope(99)
			else:
				atom.SetIsotope(0)

		ring_smarts = "[99R]@1" + "@[R]"*(size-2) + "[99R]@1"
		pat = OESubSearch()
		assert pat.Init(ring_smarts), ring_smarts
		for match in pat.Match(copy_mol):
			return set(a.GetIdx() for a in match.GetTargetAtoms())
		raise AssertionError("no ring found?")

	@staticmethod
	def get_substructure_matches(mol, pattern, uniquify=True, useChirality=False, maxMatches=1000):
		pattern.SetMaxMatches(maxMatches)
		atom_lists = []
		for match in pattern.Match(mol, uniquify):
			atom_lists.append(
				[a.GetIdx() for a in match.GetTargetAtoms()])
		return atom_lists
####

# No chemistry toolkit

no_toolkit_ruleset = Ruleset()
add_no_toolkit_rule = no_toolkit_ruleset.get_rule_decorator()

@add_no_toolkit_rule("mol", "mol_errmsg")
def get_mol():
	return None, None

@add_no_toolkit_rule()
def get_mol_atoms():
	return []

@add_no_toolkit_rule()
def get_symmetry_classes(atom_tokens):
	return [0] * len(atom_tokens)

@add_no_toolkit_rule()
def get_hcounts(atom_tokens):
	hcounts = []
	for atom in atom_tokens:
		if isinstance(atom, BracketAtomToken):
			hcount = atom.hcount
			if hcount == "H":
				hcount = 1
			else:
				hcount = int(hcount[1:])
		else:
			# XXX could try to approximate.
			hcount = 0
		hcounts.append(hcount)
	return hcounts




class no_toolkit_api:
	toolkit_name = "none"
	@staticmethod
	def get_matcher(smarts):
		raise NotImplementedError("how did I get here?")

	@staticmethod
	def find_smallest_ring(properties, from_atom_index, to_atom_index):
		raise NotImplementedError("how did I get here?")

####


def get_ruleset(toolkit="rdkit"):
	if toolkit == "rdkit":
		toolkit_ruleset = rdkit_ruleset
	elif toolkit == "openeye":
		toolkit_ruleset = openeye_ruleset
	elif toolkit == "none":
		toolkit_ruleset = no_toolkit_ruleset
	else:
		raise ValueError("Unsupported toolkit %r" % (toolkit,))

	ruleset = base_ruleset + toolkit_ruleset
	ruleset.add_namespace(
		"input",
		input_ruleset + toolkit_ruleset,
		forward_defaults = {
			"input_smiles": "smiles",
		},
	mapped_properties = {
		"input_tokens": "tokens",
		"smiles": "modified_smiles",
		"is_modified_smiles": "is_modified_smiles",
		"input_mol": "mol",
		"input_mol_errmsg": "mol_errmsg",
	})
	return ruleset

def get_toolkit_api(toolkit):
	if toolkit == "rdkit":
		return rdkit_toolkit_api
	if toolkit == "openeye":
		return openeye_toolkit_api
	return no_toolkit_api

def init_properties(smiles, toolkit="rdkit", sanitize=True,
					use_brackets=False, set_isotope="none", set_atom_class="none"):
	ruleset = get_ruleset(toolkit)
	## ruleset.dump()
	initial_values = {
		"input_smiles": smiles,
		"sanitize": True,
		"toolkit_api": get_toolkit_api(toolkit),
		"input.use_brackets": use_brackets,
		"input.set_isotope": set_isotope,
		"input.set_atom_class": set_atom_class,
		}
	properties = ruleset.get_properties(initial_values)
	properties["check_smiles_syntax"]  # May raise a ParseError

	if toolkit in ("rdkit", "openeye"):
		# Don't raise an error, but print a warning that there may be a problem.
		mol = properties["mol"]
		if mol is None:
			sys.stderr.write("Unable to generate a molecule: %s\n" % (
				properties["mol_errmsg"],))
	return properties



########### Layout


def make_layout(properties, special_symbols=UnicodeSymbols):
	smiles = properties["smiles"]
	layout = Layout(special_symbols=special_symbols)

	layout.add_above(0, smiles)
	layout.end_track("above", "SMILES")
	return layout

def _check_vertical_column(columns, column, n, num_rows):
	if column not in columns:
		return True
	elements = columns[column]
	for m in range(n, n-num_rows, -1):
		if m in elements:
			return False
	return True

# Draw vertical
def V(text, text2=None, location="above"):
	if text2 is None:
		return list(text)
	if location == "above":
		return list(text + text2)
	elif location == "below":
		return list(text2 + text)
	else:
		raise ValueError(location)


class Layout(object):
	def __init__(self, special_symbols = UnicodeSymbols):
		self.special_symbols = special_symbols

		self.columns = defaultdict(dict)
		self.ceiling = 0
		self.floor = -1

		self.legend_columns = defaultdict(dict)
		self._min_legend_row = self._max_legend_row = 0
		self._min_legend_col = self._max_legend_col = 0
		self._num_above_tracks = 0
		self._num_below_tracks = 0

	def end_track(self, location, track_label, align="center"):
		if location == "above":
			end_row = self.ceiling
			self.set_ceiling()
			start_row = self.ceiling
			if start_row == end_row:
				# Nothing present
				# Create an empty track.
				self.ceiling += 1
				start_row = self.ceiling
			start_row -= 1
			self._num_above_tracks += 1
			track_counter = self._num_above_tracks - 1

		elif location == "below":
			start_row = self.floor
			self.set_floor()
			end_row = self.floor
			if start_row == end_row:
				# Nothing present
				# Create an empty track.
				self.floor -= 1
				end_row = self.floor
			end_row += 1
			self._num_below_tracks += 1
			track_counter = self._num_below_tracks
		else:
			raise AssertionError(location)

		if align == "above":
			row = start_row
		elif align == "center":
			delta = start_row - end_row
			if delta % 2 == 1:
				delta -= 1
			delta = delta // 2
			row = start_row - delta

		elif align == "below":
			row = end_row
		else:
			raise ValueError(align)

		special_symbols = _get_special_symbols(self.special_symbols, track_counter)

		self.add_legend_text(row, -len(track_label)-1, track_label)
		if start_row == end_row:
			self.add_legend_text(start_row, -1, special_symbols.single_row)
		else:
			self.add_legend_text(start_row, -1, special_symbols.nw_corner)
			for row in range(start_row-1, end_row, -1):
				self.add_legend_text(row, -1, special_symbols.e_side)
			self.add_legend_text(end_row, -1, special_symbols.sw_corner)

	def set_level(self, location):
		if location == "above":
			self.set_ceiling()
		elif location == "below":
			self.set_floor()
		else:
			raise AssertionError(location)

	def set_ceiling(self):
		ceiling = 0
		for column in self.columns.values():
			ceiling = max(ceiling, max(column))
		if ceiling >= self.ceiling:
			self.ceiling = ceiling + 1

	def set_floor(self):
		floor = 0
		for column in self.columns.values():
			floor = min(floor, min(column))
		if floor <= self.floor:
			self.floor = floor - 1

	def find_above_row(self, start_column, end_column=None, num_rows=1):
		assert num_rows >= 1, num_rows
		if end_column is None:
			end_column = start_column + 1
		n = self.ceiling + num_rows-1  # Need enough space for all of the rows
		while 1:
			for column in range(start_column, end_column):
				if not _check_vertical_column(self.columns, column, n, num_rows):
					break
			else:
				return n
			n += 1

	def find_below_row(self, start_column, end_column=None, num_rows=1):
		assert num_rows >= 1, num_rows
		if end_column is None:
			end_column = start_column + 1
		n = self.floor
		while 1:
			for column in range(start_column, end_column):
				if not _check_vertical_column(self.columns, column, n, num_rows):
					break
			else:
				return n
			n -= 1

	def add(self, location, start_column, text, row=None):
		if location == "above":
			return self._add(text, start_column, row, self.find_above_row)
		elif location == "below":
			return self._add(text, start_column, row, self.find_below_row)
		else:
			raise ValueError(location)

	def add_above(self, start_column, text, row=None):
		return self._add(text, start_column, row, self.find_above_row)

	def add_below(self, start_column, text, row=None):
		return self._add(text, start_column, row, self.find_below_row)

	def draw_text(self, row, start_column, text):
		return self._add(text, start_column, row, None)

	def _add(self, text, start_column, row, find_row):
		if isinstance(text, type("")) or isinstance(text, type(u"")):
			lines = [text]
		else:
			lines = text
		num_rows = len(lines)
		num_cols = max(len(line) for line in lines)
		if num_rows == 0:
			return None

		if row is None:
			row = find_row(start_column, start_column+num_cols, num_rows)
		start_row = row

		for line in lines:
			for colno, c in enumerate(line, start_column):
				if not (self.floor == self.ceiling == 0):
					assert not (self.floor < row < self.ceiling), (self.floor, row, self.ceiling)
				self.columns[colno][row] = c
			row -= 1
		return (start_row, row+1)
		#print(self.columns)

	def add_legend_text(self, row, col, label):
		for i, c in enumerate(label):
			self.legend_columns[col+i][row] = c
		if row < self._min_legend_row:
			self._min_legend_row = row
		elif row > self._max_legend_row:
			self._max_legend_row = row
		if col < self._min_legend_col:
			self._min_legend_col = col
		elif col + len(label) > self._max_legend_row:
			self._max_legend_row = col + len(label)

	def get_legend_size(self):
		return (self._max_legend_col - self._min_legend_col) + 1

	def display(self, output=sys.stdout, width=40, indent=0, legend="all"):
		indent = " " * indent
		min_required_width = self.get_legend_size() + 10
		if width < min_required_width:
			raise ValueError("width of %d is too small. Must be at least %d for this layout."
								 % (width, min_required_width))
		if legend not in ("off", "once", "all"):
			raise ValueError("Unsupported legend style %r" % (legend,))
		MAX_COL = max(self.columns)

		MIN_COL = min(self.columns)
		# Find the first column containing a non-space character
		while 1:
			if MIN_COL in self.columns:
				chars = set(self.columns[MIN_COL].values())
				if chars and chars != {" "}:
					break
			if MIN_COL > MAX_COL:
				# Only spaces found?
				return
			MIN_COL += 1

		# Find the last column containing a non-space character
		while 1:
			if MAX_COL in self.columns:
				chars = set(self.columns[MAX_COL].values())
				if chars and chars != {" "}:
					break
			if MAX_COL < MIN_COL:
				raise AssertionError("Should not get here")
			MAX_COL -= 1


		legend_count = 0
		start_col = MIN_COL
		WIDTH = width
		while start_col < MAX_COL + 1:
			if legend == "all" or (legend == "once" and legend_count == 0):
				include_legend = True
				max_row = self._max_legend_row
				min_row = self._min_legend_row
				width = WIDTH - (self._max_legend_col - self._min_legend_col + 1)
			else:
				include_legend = False
				width = WIDTH
				min_row = max_row = 0
			legend_count += 1

			# Find the minimum and maximum row number for this range of columns
			for colno in range(start_col, min(start_col+width, MAX_COL+1)):
				if colno in self.columns:
					max_row = max(max_row, max(self.columns[colno]))
					min_row = min(min_row, min(self.columns[colno]))

			for rowno in range(max_row, min_row-1, -1):
				output_line = []
				if include_legend:
					for colno in range(self._min_legend_col, self._max_legend_col+1):
						if colno in self.legend_columns:
							c = self.legend_columns[colno].get(rowno, " ")
						else:
							c = " "
						output_line.append(c)

				for colno in range(start_col, min(start_col+width, MAX_COL+1)):
					if colno in self.columns:
						c = self.columns[colno].get(rowno, " ")
					else:
						c = " "
					output_line.append(c)
				output_line.append("\n")
				#print("got", output_line)
				output.write(indent + "".join(output_line))
			if colno < MAX_COL:
				# Space between this segment and the next,
				# but not after the last segment
				output.write("\n")

			start_col += width

def draw_bar(layout, start, end, start_text, end_text, c, label,
			 left_label = None, right_label=None, location="below", row=None):
	assert len(start_text) == 1, start_text
	assert len(end_text) == 1, end_text
	assert len(c) == 1, c
	if left_label is None:
		left_label = " " + label
	if right_label is None:
		right_label = label

	delta = end - start
	assert delta > 0, "cannot reverse direction"
	text = start_text + c*(delta-1) + end_text

	repeat_len = max(len(label) * 1.4, len(label) + 8)
	text_len = len(text)
	if label and text_len > repeat_len:
		inline_len = len(label) + 2
		num_terms = text_len // repeat_len
		for i in range(1, num_terms):
			offset = int(text_len * (i / float(num_terms)) - inline_len/2.0)
			text = text[:offset] + " " + label + " " + text[offset+inline_len:]

	return layout.add(location, start-len(left_label), left_label + text + right_label,
						  row=row)

#################### Tracks

class TrackManager(object):
	def __init__(self):
		self.tracks = []
		self.tracks_by_name = {}

	def __iter__(self):
		return iter(self.tracks)

	def __len__(self):
		return len(self.tracks)

	def get_track(self, track_name):
		return self.tracks_by_name[track_name]

	def add_track(self, track):
		if track.track_name in self.tracks_by_name:
			raise ValueError("Already have a track with the name %r" % (track.track_name,))
		self.tracks.append(track)
		self.tracks_by_name[track.track_name] = track

	def add_to_argparse(self, parser):
		for track in self.tracks:
			track.add_to_argparse(parser)

	def get_track_decorator(self):
		def add_track(track_name, help=None, need_toolkit=False):
			def add_track_function(func):
				local_help = help
				if local_help is None:
					# Get the help from the first non-blank line of the docstring
					local_help = inspect.getdoc(func)
					if local_help is not None:
						local_help = local_help.lstrip().splitlines()[0]

				track_args = getattr(func, "track_args", None)
				track = Track(track_name, func, track_args=track_args,
							  help=local_help, need_toolkit=need_toolkit)
				self.add_track(track)

				return func
			return add_track_function
		return add_track

def track_arg(*argparse_args, **argparse_kwargs):
	if "dest" in argparse_kwargs:
		arg_name = argparse_kwargs["dest"]
	else:
		if not argparse_args:
			raise ValueError("track_arg() must define a 'dest' kwarg or at least one non-keyword argument")
		arg_name = argparse_args[0].lstrip("-").replace("-", "_")

	if "convert" in argparse_kwargs:
		convert_func = argparse_kwargs.pop("convert")
	else:
		convert_func = None

	def track_arg_decorator(func):
		if not hasattr(func, "track_args"):
			func.track_args = []
		func.track_args.append(TrackArg(arg_name, argparse_args, argparse_kwargs, convert_func))
		return func
	return track_arg_decorator

track_manager = TrackManager()
add_track = track_manager.get_track_decorator()

class ArgumentError(Exception):
	pass
class MissingArgument(ArgumentError):
	pass
class ConvertionError(ArgumentError):
	pass

class TrackArg(object):
	def __init__(self, arg_name, argparse_args, argparse_kwargs, convert_func=None):
		argparse_kwargs = argparse_kwargs.copy()
		if "dest" in argparse_kwargs:
			dest = argparse_kwargs["dest"]
		else:
			dest = arg_name
			argparse_kwargs["dest"] = dest
		self.arg_name = arg_name
		self.argparse_args = argparse_args
		self.argparse_kwargs = argparse_kwargs.copy()
		self.convert_func = convert_func
		self.dest = dest

	def add_to_argparse(self, parser):
		parser.add_argument(*self.argparse_args, **self.argparse_kwargs)

	def get_kwarg(self, args, toolkit_api):
		value = getattr(args, self.dest)
		if self.convert_func is not None:
			return self.convert_func(value, args, self.dest, self.argparse_args[0], toolkit_api)
		else:
			return (self.dest, value)

class Track(object):
	def __init__(self, track_name, func, track_args=None, help=None, need_toolkit=False):
		if track_args is None:
			track_args = []
		self.track_name = track_name
		self.func = func
		self.track_args = track_args

		self.help = help
		self.need_toolkit = need_toolkit

	def add_to_argparse(self, parser):
		if self.track_args:
			group = parser.add_argument_group("Options for the %r track" % (self.track_name,))
			for track_args in self.track_args:
				track_args.add_to_argparse(group)

	def get_kwargs(self, args, toolkit_api):
		kwargs = {}
		for track_arg in self.track_args:
			pair = track_arg.get_kwarg(args, toolkit_api)
			if pair is not None:
				k, v = pair
				kwargs[k] = v
		return kwargs

	def add_to_layout(self, layout, properties, location="bottom", **kwargs):
		self.func(layout, properties, location=location, **kwargs)


@add_track("offsets", "display the offset of every 5th byte in the SMILES string, and the last byte")
def add_byte_offset_track(layout, properties, location):
	"Display the offset of every 5th byte in the SMILES string, and the last byte"
	n = len(properties["smiles"])
	for i in range(0, n-1, 5):
		if i % 5 == 0:
			layout.add(location, i, V(str(i)))
	layout.add(location, n-1, V(str(n-1)))
	layout.end_track(location, "byte offsets")

@add_track("atoms", help="display the index number of each atom term")
def add_atom_index_track(layout, properties, location):
	for atom_index, atom in enumerate(properties["atom_tokens"]):
		layout.add(location, atom.symbol_start, V(str(atom_index), "|", location))
	layout.end_track(location, "atoms")


@add_track("input-smiles", "show the input SMILES, before any processing, aligned with the main SMILES")
def add_input_smiles_track(layout, properties, location):
	"Display the input SMILES (before any SMILES processing)"
	input_tokens = properties["input_tokens"]
	tokens = properties["tokens"]
	assert len(input_tokens) == len(tokens), (len(input_tokens), len(tokens))
	for input_token, token in zip(input_tokens, tokens):
		if isinstance(input_token, AtomToken):
			input_term = input_token.term
			input_delta = input_token.symbol_start - input_token.start
			# XXX what about showing middle dots for the inserted characters?
			start_row, end_row = layout.add(location, token.symbol_start - input_delta, input_term)
		else:
			layout.add(location, token.start, input_token.term)

	layout.end_track(location, "input smiles")

_state_aliases = {
	"atom": ("A", "a"),
	"bond": ("B", "b"),
	"open_branch": ("(", "("),
	"close_branch": (")", ")"),
	"dot": (".", "."),
	"closure": ("%", "d"),
	}
@add_track("tokens", "display the index number of each term")
def add_token_label_track(layout, properties, location):
	for token in properties["tokens"]:
		start_char, rest_char = _state_aliases[token.typename]
		c = start_char
		for col in range(token.start, token.end):
			layout.add(location, col, c)
			c = rest_char
	layout.end_track(location, "token types")

@add_track("hcounts", "show the implicit hydrogen count on each atom", need_toolkit=True)
def add_hcount_track(layout, properties, location):
	mol = properties["mol"]
	if mol is None:
		layout.add(location, 0, "(no molecule available)")
	else:
		hcounts = properties["hcounts"]
		for atom_token, hcount in zip(properties["atom_tokens"], hcounts):
			layout.add(location, atom_token.symbol_start, V(str(hcount), location=location))
	layout.end_track(location, "hcounts")


@add_track("branches", help="show the start and end location of each pair of branches")
def add_branch_track(layout, properties, location):
	tokens = properties["tokens"]
	base_atoms = defaultdict(list)
	for branch in properties["branches"]:
		base_atoms[branch.base_atom_token_index].append(branch)

	for atom_token_index, branches in sorted(base_atoms.items()):
		atom_token = tokens[atom_token_index]

		# Find out where I can place this section
		# I have a leading space to keep things from bunching up.
		spaces = " " * (tokens[branches[-1].close_token_index].end - atom_token.start)
		start_row, end_row = layout.add(location, atom_token.start-1, spaces)

		gap = (tokens[branches[0].open_token_index].start - atom_token.start) - 1
		layout.draw_text(start_row, atom_token.symbol_start, "*" + "-"*gap)
		label = str(atom_token.atom_index)
		for branch in branches:
			draw_bar(layout,
					 tokens[branch.open_token_index].start,
					 tokens[branch.close_token_index].end-1,
					 layout.special_symbols.open_branch,
					 layout.special_symbols.close_branch,
					 layout.special_symbols.in_branch,
					 label, "", "",
					 row = start_row)

	layout.end_track(location, "branches")

@add_track("closures", help="show the start and end location of each pair of closures",
		   need_toolkit=True)
@track_arg(
	"--closure-atom-style",
	choices=("default", "atoms", "elements", "end-atoms", "end-elements",
			 "end-atoms-only", "end-elements-only", "none"),
	default="default",
	help=(
		"The 'atoms' style indicates the location atom with a '*'. "
		"The 'end-atoms' style indicates location of the ends of the closure with a '*' "
		"and the other atoms with an 'x'. The 'end-atoms-only' style only indicates the "
		"end atom locations. The '*-elements' variants show the start location of the atomic "
		"element rather than the full atom location. "
		"Use 'none' to not display atom locations. "
		"(default: 'atoms')")
		)
@track_arg(
	"--closure-style",
	choices=("default", "arrows", "text", "none"),
	default="default",
	help="The default of 'arrow' uses an up-arrow to indicate the closure location. "
		 "The 'text' style shows the closure text. "
		 "Use 'none' to not indicate the closure location. "
		 "(default: 'text')"
	)
def add_closure_track(layout, properties, location, closure_atom_style="default", closure_style="default"):
	if properties["mol"] is None:
		layout.add(location, 0, "(no molecule available)")
		layout.end_track(location, "closures")
		return

	tokens = properties["tokens"]
	closures = properties["closures"]

	# Reorder them by the first closure location, not last
	closures = sorted(closures, key=lambda closure: closure.first_closure)

	toolkit_api = properties["toolkit_api"]
	#atom_rings = [set(atom_indices) for atom_indices in properties["mol"].GetRingInfo().AtomRings()]

	background_label = "."

	if closure_atom_style == "default":
		closure_atom_style = "atoms"

	if closure_atom_style == "atoms":
		# Highlight all of every atom
		closure_atom_label = layout.special_symbols.closure_atom
		closure_other_atom_label = layout.special_symbols.closure_atom
		show_complete_atom = True
	elif closure_atom_style == "elements":
		# Highlight just the symbol start location
		closure_atom_label = layout.special_symbols.closure_atom
		closure_other_atom_label = layout.special_symbols.closure_atom
		show_complete_atom = False
	elif closure_atom_style == "end-atoms":
		closure_atom_label = layout.special_symbols.closure_atom
		closure_other_atom_label = layout.special_symbols.closure_other_atoms
		show_complete_atom = True
	elif closure_atom_style == "end-elements":
		closure_atom_label = layout.special_symbols.closure_atom
		closure_other_atom_label = layout.special_symbols.closure_other_atoms
		show_complete_atom = False
	elif closure_atom_style == "end-atoms-only":
		closure_atom_label = layout.special_symbols.closure_atom
		closure_other_atom_label = background_label
		show_complete_atom = True
	elif closure_atom_style == "end-elements-only":
		closure_atom_label = layout.special_symbols.closure_atom
		closure_other_atom_label = background_label
		show_complete_atom = False
	elif closure_atom_style == "none":
		closure_atom_label = background_label
		closure_other_atom_label = background_label
		show_complete_atom = False
	else:
		raise ValueError(closure_atom_style)

	if closure_style == "default":
		closure_style = "text"

	if closure_style == "arrows":
		closure_label = layout.special_symbols.closure_label
	elif closure_style == "text":
		closure_label = None
	elif closure_style == "none":
		closure_label = background_label
	else:
		raise ValueError(closure_style)

	assert len(closure_atom_label) == 1, closure_atom_label
	assert len(closure_other_atom_label) == 1, closure_other_atom_label
	assert closure_label is None or len(closure_label) == 1, closure_label

	for closure in closures:
		# Find the ring
		from_atom_index = tokens[closure.first_atom].atom_index
		to_atom_index = tokens[closure.second_atom].atom_index

		atom_indices = toolkit_api.find_smallest_ring(properties, from_atom_index, to_atom_index)
		## for atom_indices in atom_rings:
		##     if from_atom_index in atom_indices and to_atom_index in atom_indices:
		##         break
		## else:
		##     atom_indices = {from_atom_index, to_atom_index}

		start_row, end_row = draw_subgraph(
			layout, properties, atom_indices,
			label = background_label,
			atom_symbol_label = closure_other_atom_label,
			atom_label = (closure_other_atom_label if show_complete_atom else background_label),
			use_full_row=False, location=location)

		# Mark the closure symbols
		for token_index in (closure.first_closure, closure.second_closure):
			token = tokens[token_index]

			start = token.start
			if closure_label is None:
				text = token.term
			else:
				text = closure_label
				term = token.term
				# Find the first digit
				if term[:1] == "%":
					if term[:2] == "%(":
						start += 2
					else:
						start += 1

			layout.draw_text(start_row, start, text)

		# Mark the end atoms
		for token_index in (closure.first_atom,
							closure.second_atom):
			token = tokens[token_index]
			if show_complete_atom:
				for col in range(token.start, token.end):
					layout.draw_text(
						start_row,
						col,
						closure_atom_label)
			else:
				layout.draw_text(
					start_row,
					token.symbol_start,
					closure_atom_label)

	layout.end_track(location, "closures")

@add_track("smiles", "display another copy of the SMILES")
def add_smiles_track(layout, properties, location="below"):
	layout.add(location, 0, properties["smiles"])
	layout.end_track(location, "SMILES")

def convert_smarts(smarts, args, dest, flag, toolkit_api):
	if smarts is None:
		raise MissingArgument("Must specify %s" % (flag,))

	matcher = toolkit_api.get_matcher(smarts)
	if matcher is None:
		raise ArgumentError("The %s toolkit cannot process %s" % (
			toolkit_api.toolkit_name, flag))
	return ("pattern", matcher)


@add_track("matches", "show which atoms match a given SMARTS match (--smarts is required)")
@track_arg(
	"--smarts", convert=convert_smarts,
	help="SMARTS pattern to use for the 'matches' track(s)")
@track_arg(
	"--all-matches", dest="uniquify", action="store_false",
	help="Show all matches. The default only shows unique matches.")
@track_arg(
	"--max-matches", dest="maxMatches", type=int, metavar="N", default=1000,
	help="The maximum number of matches to display. (default: 1000)")
@track_arg(
	"--use-chirality", dest="useChirality", action="store_true",
	help="Enable the use of stereochemistry during matching.")
@track_arg(
	"--match-style", choices=("simple", "pattern-index", "atom-index"),
	help="Change the display style from a simple '*' to something which also shows "
		 "the pattern or atom index")
def add_smarts_match_tracks(layout, properties, location,
							pattern, uniquify=True, useChirality=False,
							maxMatches=1000, match_style=None,
							):

	mol = properties["mol"]
	matches = properties["toolkit_api"].get_substructure_matches(
		mol, pattern, uniquify=uniquify, useChirality=useChirality, maxMatches=maxMatches)

	if match_style == "simple" or match_style is None:
		def style(atom_index, pattern_index):
			return "*"
	elif match_style == "pattern-index":
		def style(atom_index, pattern_index):
			return V(str(pattern_index))
	elif match_style == "atom-index":
		def style(atom_index, pattern_index):
			return V(str(atom_index))
	else:
		raise ValueError("Unknown match_style: %r" % (match_style,))


	atom_tokens = properties["atom_tokens"]
	for matchno, atom_indices in enumerate(matches, 1):
		for pattern_index, atom_index in enumerate(atom_indices):
			layout.add(location, atom_tokens[atom_index].symbol_start, style(atom_index, pattern_index))
		layout.end_track(location, "match %d" % (matchno,))

_bond_type_labels = None
def _init_bond_type_labels():
	global _bond_type_labels
	from rdkit import Chem
	_bond_type_labels = {
		Chem.BondType.SINGLE: "-",
		Chem.BondType.DOUBLE: "=",
		Chem.BondType.AROMATIC: ":",
		Chem.BondType.TRIPLE: "#",
		}

@add_track("neighbors", "show which atoms are connected to a given atom index (--atom-index is required)")
@track_arg("--atom-index", "--idx", type=int, metavar="N",
		   help="Define the atom to use for the 'neighbors' track.")
def add_neighbor_track(layout, properties, location, atom_index):
	if atom_index is None:
		return  # ignore unless --atom-index is specified

	atom_tokens = properties["atom_tokens"]
	mol = properties["mol"]
	if atom_index >= len(atom_tokens):
		layout.add(location, 0, "No atom with index %d" % (atom_index,))
	else:
		mol_graph = properties["mol_graph"]
		#center_atom = mol.GetAtomWithIdx(atom_index)
		center_atom = mol_graph.atoms[atom_index]
		#start = atom_tokens[atom_index].symbol_start
		start = center_atom.token.symbol_start

		layout.add(location, start, "*")

		terms = []
		for bond_idx, bond_label, other_atom_idx in center_atom.get_outgoing():
			terms.append("%s%s%d" % (bond_label,
									 mol_graph.atoms[other_atom_idx].atom_symbol,
									 other_atom_idx))

		description = center_atom.atom_symbol + "".join("(" + term + ")" for term in terms)

		layout.add(location, start, description)

		arrow = layout.special_symbols.towards_arrows[location]
		for bond_idx, bond_label, other_atom_idx in center_atom.get_outgoing():
			other_atom_start = atom_tokens[other_atom_idx].symbol_start
			layout.add(location, other_atom_start, arrow)

	layout.end_track(location, "neighbors")

def draw_subgraph(layout, properties, atom_indices,
				  label = None,
				  atom_label = None,
				  atom_symbol_label = None,
				  bond_label = None,
				  branch_label = None,
				  closure_label = None,
				  use_full_row=True, location="below"):
	tokens = properties["tokens"]
	atom_tokens = properties["atom_tokens"]

	if label is None:
		label = "-"

	if atom_label is None:
		atom_label = label
	if atom_symbol_label is None:
		atom_symbol_label = label
	if bond_label is None:
		bond_label = label
	if branch_label is None:
		branch_label = label
	if closure_label is None:
		closure_label = label

	label_size = max(len(atom_label), len(atom_symbol_label), len(bond_label),
					 len(branch_label), len(closure_label))
	atom_label = V(atom_label)
	atom_symbol_label = V(atom_symbol_label)
	bond_label = V(bond_label)
	branch_label = V(branch_label)
	closure_label = V(closure_label)

	if not (label_size > 0):
		raise ValueError("One of the labels must be non-empty")

	# Record the text commands so I can find the min/max output bounds
	label_locations = {}
	def draw_label_at(i, label):
		label_locations[i] = label

	# Mark the atoms
	for atom_index in atom_indices:
		atom_token = atom_tokens[atom_index]
		for i in range(atom_token.start, atom_token.end):
			if i == atom_token.symbol_start:
				draw_label_at(i, atom_symbol_label)
			else:
				draw_label_at(i, atom_label)

	atom_indices = set(atom_indices)

	# Mark the branches if both atoms are selected
	for branch in properties["branches"]:
		from_atom_index = tokens[branch.base_atom_token_index].atom_index
		to_atom_index = tokens[branch.first_branch_atom_token_index].atom_index
		if from_atom_index in atom_indices and to_atom_index in atom_indices:
			# Both atoms are selected, so mark the branch
			# The token is either '(' or ')' so there is only one column to draw.
			for token_index in (branch.open_token_index, branch.close_token_index):
				draw_label_at(tokens[token_index].start, branch_label)

	# Mark the closures if both atoms are selected
	for closure in properties["closures"]:
		from_atom_index = tokens[closure.first_atom].atom_index
		to_atom_index = tokens[closure.second_atom].atom_index
		if from_atom_index in atom_indices and to_atom_index in atom_indices:
			for index in (closure.first_closure, closure.second_closure):
				for i in range(tokens[index].start, tokens[index].end):
					draw_label_at(i, closure_label)

	# And the bonds
	for graph_bond in properties["graph"].bonds: # XXX use "graph_bonds" instead?
		if graph_bond.token is None:
			# implicit bond; not present in the tokens
			continue
		from_atom_index, to_atom_index = graph_bond.atom_indices
		if from_atom_index in atom_indices and to_atom_index in atom_indices:
			for i in range(graph_bond.token.start, graph_bond.token.end):
				draw_label_at(i, bond_label)

	# Now I know where to draw all of the labels.
	if use_full_row:
		s = [" " * len(properties["smiles"])] * label_size
		start_column = 0
	else:
		min_column = min(label_locations)
		max_column = max(label_locations)
		num_columns = max_column - min_column + 1

		# Require two spaces between displays (space before and after)
		start_column = min_column-1
		s = [" " * (num_columns + 2)] * label_size

	# Find the space for the output
	start_row, end_row = layout.add(location, start_column, s)
	# Write the labels
	for col, v_label in label_locations.items():
		layout.draw_text(start_row, col, v_label)

	return start_row, end_row

@add_track("fragments", "show which atoms are in each connected fragment")
def add_fragment_track(layout, properties, location):
	tokens = properties["tokens"]
	atom_tokens = properties["atom_tokens"]
	fragments = properties["fragments"]
	for fragment_index, atom_indices in enumerate(fragments):
		draw_subgraph(layout, properties, atom_indices,
					  label = str(fragment_index),
					  location = location)

	layout.end_track(location, "fragments")


@add_track("symclasses", help="show the atom symmetry classes", need_toolkit=True)
def add_hcount_track(layout, properties, location):
	mol = properties["mol"]
	if mol is None:
		layout.add(location, 0, "(no molecule available)")
	else:
		hcounts = properties["symmetry_classes"]
		for atom_token, hcount in zip(properties["atom_tokens"], hcounts):
			layout.add(location, atom_token.symbol_start, V(str(hcount), location=location))
	layout.end_track(location, "symclasses")


@add_track("none", help="show nothing")
def add_none_track(layout, properties, location):
	return

########### Command-line processing

track_names = [track.track_name for track in track_manager] + [
		"default", "fancy"]

TRACKS = "".join("%s - %s\n" % (track.track_name, track.help) for track in track_manager)
TRACKS += (
	"default - the default tracks for the given input\n"
	"fancy - show most of the relevant tracks\n"
	)

epilog = """
The available tracks are:
""" + "".join("  " + line for line in TRACKS.splitlines(True))

epilog += """

If no --above tracks are specified then the default shows the atom
indices. If one of the input-modifying options (like --use-brackets)
is used, then the "input-smiles" track will also be shown.

If no --below tracks are specified then the default ... well, it's
complicated, and I think the current logic is buggy. It tries to show
the useful tracks for what you asked for.

Use the "--fancy" option to have smiview show more tracks than the
default.

To disable track display, use "-a none -b none". This tells smiview to
not use the default tracks but only to show the "none" tracks, which
does nothing. For example:
  %(prog)s 'CCO' -a none --use-rdkit
will only verify the syntax and display the SMILES string

Examples:

  %(prog)s 'Cc1c(OC)c(C)cnc1CS(=O)c2nc3ccc(OC)cc3n2' --fancy
  %(prog)s 'O/N=C/5C.F5' -a offsets -b closures
  %(prog)s 'CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO' --smarts '[R]'
  %(prog)s 'CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21' --atom-index 2
"""


class ListTracksAction(argparse.Action):
	def __init__(self, option_strings, dest, nargs=None, **kwargs):
		if nargs is not None:
			raise ValueError("nargs not allowed")
		super(ListTracksAction, self).__init__(option_strings, dest, nargs=0, **kwargs)
	def __call__(self, parser, namespace, values, option_string=None):
		sys.stdout.write(TRACKS)
		raise SystemExit(0)

class FancyAction(argparse.Action):
	def __init__(self, option_strings, dest, nargs=None, **kwargs):
		if nargs is not None:
			raise ValueError("nargs not allowed")
		super(FancyAction, self).__init__(option_strings, dest, nargs=0, **kwargs)
	def __call__(self, parser, namespace, values, option_string=None):
		if getattr(namespace, "above", None) is None:
			namespace.above = []
		namespace.above.append("fancy")
		if getattr(namespace, "below", None) is None:
			namespace.below = []
		namespace.below.append("fancy")

_kwargs = dict(
	description = "Show details of the SMILES string",
	allow_abbrev = False,
	epilog = epilog,
	formatter_class = argparse.RawDescriptionHelpFormatter,
	)
if sys.version_info[0] == 2:
	_kwargs.pop("allow_abbrev") # Not available in older Pythons

parser = argparse.ArgumentParser(**_kwargs)
del _kwargs

parser.add_argument(
	"--list-tracks", "-l", action=ListTracksAction,
	help="List the available tracks.")

parser.add_argument(
	"--above", "-a", action="append", metavar="TRACK", choices=track_names,
	help="Specify a track to show above the SMILES. Repeat this option once for each track.")
parser.add_argument(
	"--below", "-b", action="append", metavar="TRACK", choices=track_names,
	help="Specify a track to show below the SMILES. Repeat this option once for each track.")
parser.add_argument(
	"--fancy", action=FancyAction,
	help="alias for '--above fancy --below fancy'")
parser.add_argument(
	"--toolkit", choices=("rdkit", "openeye", "auto", "none"), default="auto",
	help="Specify which chemistry toolkit to use.") # XXX Finish docs


track_manager.add_to_argparse(parser)

input_group = parser.add_argument_group("Input modification options")
input_group.add_argument(
	"--use-brackets", action="store_true",
	help="Modify the input SMILES so the atoms in the organic subset are now in brackets. "
		 "Use a chemistry toolkit to get the correct hydrogen counts, otherwise the count will be 0.")
input_group.add_argument(
	"--set-isotope", choices=("none",) + tuple(sorted(_sources)), default="none",
	help="same as --use-brackets followed by setting the isotope field of each atom to the specified value")
input_group.add_argument(
	"--set-atom-class", choices=("none",) + tuple(sorted(_sources)), default="none",
	help="same as --use-brackets followed by setting the atom class field of each atom to the specified value")

rdkit_group = parser.add_argument_group("RDKit processing options")
rdkit_group.add_argument(
	"--no-sanitize", action="store_true",
	help="Do not let RDKit sanitize/modify the bond orders and charges")

output_group = parser.add_argument_group("Output formatting options")
output_group.add_argument(
	"--width", type=int, default=72, metavar="W",
	help="Number of columns to use in the output. Must be at least 40. (default: 72)")
output_group.add_argument(
	"--indent", type=int, default=0, metavar="N",
	help="Indent the output by N spaces. Does not affect the width. (default: 0)")

output_group.add_argument(
	"--legend", choices=("off", "once", "all"), default="all",
	help="The default of 'all' shows the legend for each output segment. "
	"Use 'once' to only show it in the first segment, or 'off' for no legend.")

output_group.add_argument(
	"--ascii", action="store_true",
	help="Use pure ASCII for the output, instead of Unicode characters")

output_group.add_argument(
	"--encoding", default="utf8",
	help="specify the output encoding (default: utf8)")

parser.add_argument(
	"--version", action='version', version='%(prog)s ' + __version__)

parser.add_argument(
	"smiles", metavar="SMILES", nargs="?",
	help = "SMILES string to show (if not specified, use caffeine)"
	)

def die(msg):
	sys.stderr.write(msg)
	if not msg.endswith("\n"):
		sys.stderr.write("\n")
	sys.stderr.flush()
	raise SystemExit(1)

_default_above = ["atoms"]
_default_below = ["hcounts", "branches", "closures", "fragments"]
_fancy_above = ["offsets", "atoms", "tokens"]
_fancy_below = ["hcounts", "branches", "closures", "fragments", "symclasses"]


_track_defaults = {
	"above": {
		"default-simple": _default_above,
		"default-change-input": _default_above + ["input-smiles"],
		"default-smarts": _default_above,
		"default-atom-index": _default_above,

		"fancy-simple": _fancy_above,
		"fancy-change-input": _fancy_above + ["input-smiles"],
		"fancy-smarts": _fancy_above,
		"fancy-atom-index": _fancy_above,
		},

	"below": {
		"default-simple": _default_below,
		"default-change-input": _default_below,
		"default-smarts": ["matches"],
		"default-atom-index": ["neighbors"],

		"fancy-simple": _fancy_below,
		"fancy-change-input": _fancy_below,
		"fancy-smarts": _fancy_below + ["matches"],
		"fancy-atom-index": _fancy_below + ["neighbors"],
		}
	}

def _merge_aliases(alias_names, default_table):
	seen = set()
	merged_names = []
	for alias in alias_names:
		for name in default_table[alias]:
			if name not in seen:
				seen.add(name)
				merged_names.append(name)
	return merged_names

def _get_alias_track_names(alias_prefix, alias_table, change_input, use_smarts, use_atom_index):
	if not (use_smarts or use_atom_index or change_input):
		alias_names = [alias_prefix + "-simple"]
	else:
		alias_names = []
		if change_input:
			alias_names.append(alias_prefix + "-change-input")
		if use_smarts:
			alias_names.append(alias_prefix + "-smarts")
		if use_atom_index:
			alias_names.append(alias_prefix + "-atom-index")
	return _merge_aliases(alias_names, alias_table)

def get_tracks_and_kwargs(input_tracks, track_manager, args, location, toolkit_api,
						  change_input, use_smarts, use_atom_index, has_toolkit):
	alias_table = _track_defaults[location]
	default_names = _get_alias_track_names(
		"default", alias_table, change_input, use_smarts, use_atom_index)
	if not has_toolkit:
		default_names = [name for name in default_names if not track_manager.get_track(name).need_toolkit]

	fancy_names = _get_alias_track_names(
		"fancy", alias_table, change_input, use_smarts, use_atom_index)
	if not has_toolkit:
		fancy_names = [name for name in fancy_names if not track_manager.get_track(name).need_toolkit]

	if input_tracks is None:
		track_names = default_names
	else:
		track_names = []
		for name in input_tracks:
			if name == "default":
				track_names.extend(default_names)
			elif name == "fancy":
				track_names.extend(fancy_names)
			else:
				track_names.append(name)

	tracks_and_kwargs = []
	for track_name in track_names:
		track = track_manager.get_track(track_name)
		try:
			kwargs = track.get_kwargs(args, toolkit_api)
		except ArgumentError as err:
		   die("Cannot use track %r: %s" % (track_name, err))
		tracks_and_kwargs.append( (track, kwargs) )

	return tracks_and_kwargs


def smiview_atom_idx(smiles, toolkit = "auto"):
	args = parser.parse_args("")

	if smiles is None:
		sys.stderr.write("No SMILES specified. Using caffeine.\n")
		smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

	if toolkit == "auto":
		if has_rdkit():
			toolkit = "rdkit"
		elif has_openeye():
			toolkit = "openeye"
		else:
			toolkit = "none"
	toolkit_api = get_toolkit_api(toolkit)

	change_input = (
		args.use_brackets or
		args.set_isotope != "none" or
		args.set_atom_class != "none"
		)
	use_smarts = args.smarts is not None
	use_atom_index = args.atom_index is not None
	has_toolkit = toolkit != "none"
	above_tracks_and_kwargs = get_tracks_and_kwargs(
		args.above, track_manager, args, "above", toolkit_api,
		change_input, use_smarts, use_atom_index,
		has_toolkit)

	below_tracks_and_kwargs = get_tracks_and_kwargs(
		args.below, track_manager, args, "below", toolkit_api,
		change_input, use_smarts, use_atom_index,
		has_toolkit)

	try:
		properties = init_properties(
			smiles, toolkit=toolkit,
			use_brackets = args.use_brackets,
			set_isotope = args.set_isotope,
			set_atom_class = args.set_atom_class,
			)
	except ParseError as err:
		die(err.get_report("Cannot parse --smiles: "))

	idx_atom = []
	for atom_index, atom in enumerate(properties["atom_tokens"]):
		idx_atom.append(atom.symbol_start)

	return idx_atom



def main(argv=None):
	args = parser.parse_args(argv)

	smiles = args.smiles
	if smiles is None:
		sys.stderr.write("No SMILES specified. Using caffeine.\n")
		smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

	toolkit = args.toolkit
	if toolkit == "auto":
		if has_rdkit():
			toolkit = "rdkit"
		elif has_openeye():
			toolkit = "openeye"
		else:
			toolkit = "none"
	toolkit_api = get_toolkit_api(toolkit)

	change_input = (
		args.use_brackets or
		args.set_isotope != "none" or
		args.set_atom_class != "none"
		)
	use_smarts = args.smarts is not None
	use_atom_index = args.atom_index is not None
	has_toolkit = toolkit != "none"
	above_tracks_and_kwargs = get_tracks_and_kwargs(
		args.above, track_manager, args, "above", toolkit_api,
		change_input, use_smarts, use_atom_index,
		has_toolkit)

	below_tracks_and_kwargs = get_tracks_and_kwargs(
		args.below, track_manager, args, "below", toolkit_api,
		change_input, use_smarts, use_atom_index,
		has_toolkit)

	try:
		properties = init_properties(
			smiles, toolkit=toolkit,
			use_brackets = args.use_brackets,
			set_isotope = args.set_isotope,
			set_atom_class = args.set_atom_class,
			)
	except ParseError as err:
		die(err.get_report("Cannot parse --smiles: "))
	if args.ascii:
		special_symbols = ASCIISymbols
		encoding = "ascii"
	else:
		special_symbols = UnicodeSymbols
		encoding = args.encoding
		try:
			"".encode(encoding)
		except Exception as err:
			die("Error with --encoding: %s" % (err,))
		try:
			u"\U00002502".encode(encoding)
		except Exception as err:
			die("--encoding %r does not support enough of Unicode" % (encoding,))


	width = args.width
	if width < 40:
		die("--width is too narrow. Must be at least 40.")
	indent = args.indent
	if indent < 0:
		die("--indent must be non-negative")

	layout = make_layout(properties, special_symbols=special_symbols)

	for location, tracks_and_kwargs in (
			("above", above_tracks_and_kwargs[::-1]),
			("below", below_tracks_and_kwargs),
			):
		for track, kwargs in tracks_and_kwargs:
			track.add_to_layout(layout, properties, location=location, **kwargs)

	stdout = sys.stdout
	# This is the way I know to get to the byte output stream under
	# both Python 2.7 and Python 3.
	stdout = getattr(sys.stdout, "buffer", stdout)
	unicode_stdout = codecs.getwriter(encoding)(stdout)
	layout.display(output=unicode_stdout, width=width, indent=indent, legend=args.legend)

if __name__ == "__main__":
	main()