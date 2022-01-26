#include "compiler.h"

#include "minipy/common/instruction.h"
#include "minipy/compiler/Serialization.h"
#include "minipy/compiler/symbol_table.h"
#include "minipy/interpreter/types.h"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <stdexcept>
#include <unordered_set>

namespace minipy {
namespace {

struct DictKeyHash {
  size_t operator()(const Obj& obj) const {
    // TODO
    if (obj.isInt()) {
      return std::hash<int64_t>()(obj.toInt());
    } else if (obj.isDouble()) {
      return std::hash<double>()(obj.toDouble());
    } else if (obj.isBool()) {
      return std::hash<bool>()(obj.toBool());
    } else if (obj.isDynamic()) {
      return std::hash<c10::intrusive_ptr<Dynamic>>()(obj.toDynamic());
    } else if (obj.isNone()) {
      return std::hash<int64_t>()(0);
    } else if (obj.isString()) {
      return std::hash<std::string>()(obj.toStringRef());
    } else {
      throw std::runtime_error(
          "Can't hash IValues with tag '" + obj.typeName() + "'");
    }
  }
};

enum class NameContext { Load, Store, Del, AugLoad, AugStore, Param };

OpCode getNameOp(
    SymbolTableEntry::BlockKind blockKind,
    SymbolScope scope,
    NameContext context) {
  switch (scope) {
    case SymbolScope::GLOBAL: {
      switch (context) {
        case NameContext::Load:
          return OpCode::LOAD_GLOBAL;
        case NameContext::Store:
          return OpCode::STORE_GLOBAL;
        case NameContext::AugStore:
        case NameContext::AugLoad:
        case NameContext::Del:
        case NameContext::Param:
          break;
      }
    }
    case SymbolScope::LOCAL: {
      switch (context) {
        case NameContext::Load:
          return blockKind == SymbolTableEntry::BlockKind::Function
              ? OpCode::LOAD_FAST
              : OpCode::LOAD_NAME;
        case NameContext::Store:
          return blockKind == SymbolTableEntry::BlockKind::Function
              ? OpCode::STORE_FAST
              : OpCode::STORE_NAME;
        case NameContext::AugStore:
        case NameContext::AugLoad:
        case NameContext::Del:
        case NameContext::Param:
          break;
      }
    }
  }
  assert(false);
}
struct BasicBlock;

// Represents an raw instruction + metadata for the compiler
struct CompilerInstruction {
  /* implicit */ CompilerInstruction(Instruction inst)
      : inst_(std::move(inst)), jumpTarget_(nullptr) {}
  CompilerInstruction(Instruction inst, BasicBlock* jumpTarget)
      : inst_(std::move(inst)), jumpTarget_(jumpTarget) {
    assert(jumpTarget_);
    // // TODO Should only be jump instr
    // TORCH_INTERNAL_ASSERT(
    //     inst_.op == OpCode::POP_JUMP_IF_FALSE ||
    //     inst_.op == OpCode::JUMP_IF_FALSE_OR_POP ||
    //     inst_.op == OpCode::JUMP_IF_TRUE_OR_POP ||
    //     inst_.op == OpCode::FO ||
    //     inst_.op == OpCode::JUMP_ABSOLUTE);
  }

  Instruction inst_;
  BasicBlock* jumpTarget_;
};

/*
 * Structure of compilation:
 * - Compiler: whole program
 *   - CompilerUnit: a "scope" (Functions, Classes, Modules)
 *     - BasicBlock: a straight-line sequence of instructions (ifs, loops)
 *
 * Parallel structure for symtable:
 * -  SymbolTable <-> Compiler
 *   - SymbolTableEntry <-> CompilerUnit
 *
 * Assembler: takes a CompilerUnit and produces a CodeObject.
 */

static void indent(size_t level) {
  fmt::print("{}", std::string(' ', level));
}

// Represents a straight-line bytecode sequence. Ifs, loops, etc. generate new
// basic blocks
struct BasicBlock {
  std::vector<CompilerInstruction> instructions_;
  BasicBlock* next_ = nullptr;
  bool hasReturn = false;
  size_t offset;

  void dump(size_t indentLevel = 0) const {
    // indent(indentLevel);

    // fmt::print("BasicBlock ({})\n", (void*)this);
    // for (const auto& instruction : instructions_) {
    //   indent(indentLevel);
    //   if (instruction.jumpTarget_) {
    //     fmt::print(
    //         "{} (target: {})\n",
    //         instruction.inst_,
    //         (void*)instruction.jumpTarget_);
    //   } else {
    //     fmt::print("{}\n", instruction.inst_);
    //   }
    // }
    // fmt::print("\n");
  }
};

// Represents an individual execution unit (something that the interpreter can
// run): Functions, Classes, and Modules are units.
class CompilerUnit {
 public:
  explicit CompilerUnit(SymbolTableEntry* ste) : symbolTableEntry_(ste) {
    // Add function parameters to local variable names
    size_t i = 0;
    for (const auto& argName : symbolTableEntry_->args) {
      varnames_[argName] = i++;
    }
    curBlock_ = addBlock();
  }

  void dump() const {
    BasicBlock* cur = basicBlocks_.at(0).get();
    cur->dump();
    while (cur->next_) {
      cur = cur->next_;
      cur->dump();
    }
  }

  BasicBlock* addBlock() {
    basicBlocks_.push_back(std::make_unique<BasicBlock>());
    return basicBlocks_.back().get();
  }

  void enterBlock(BasicBlock* b) {
    curBlock_->next_ = b;
    curBlock_ = b;
  }

  // BasicBlock* pop() {
  //   TORCH_INTERNAL_ASSERT(!stack_.empty());
  //   BasicBlock* back = stack_.back();
  //   stack_.pop_back();
  //   if (stack_.empty()) {
  //     curBlock_ = nullptr;
  //   } else {
  //     curBlock_ = stack_.back();
  //   }
  //   return back;
  // }
  void emitNameOp(const Ident ident, NameContext context) {
    const auto& name = ident.name();
    auto it = symbolTableEntry_->symbols.find(name);
    if (it == symbolTableEntry_->symbols.end()) {
      throw ErrorReport(ident.range()) << "name not found in symbol table";
    }
    SymbolScope scope = it->second.getScope();
    // Add this to the names table
    OpCode op = getNameOp(symbolTableEntry_->kind, scope, context);

    // In addition to the basic names table, we should also add this to the fast
    // lookup table
    size_t idx;
    if (op == OpCode::STORE_FAST || op == OpCode::LOAD_FAST) {
      idx = add(name, varnames_);
    } else {
      idx = add(name, names_);
    }
    addInstruction(Instruction(op, idx));
  }

  size_t addName(const std::string& name) {
    return add(name, names_);
  }

  size_t addConst(Obj const_) {
    return add(const_, consts_);
  }

  void addInstruction(Instruction instruction) {
    if (instruction.op == OpCode::RETURN_VALUE) {
      curBlock_->hasReturn = true;
    }
    curBlock_->instructions_.emplace_back(instruction);
  }

  // TODO consolidate with above
  void addInstruction(Instruction instruction, BasicBlock* jumpTarget) {
    if (instruction.op == OpCode::RETURN_VALUE) {
      curBlock_->hasReturn = true;
    }
    curBlock_->instructions_.emplace_back(instruction, jumpTarget);
  }

 private:
  template <typename T, typename Hash, typename EqualTo>
  size_t add(const T& arg, std::unordered_map<T, size_t, Hash, EqualTo>& map) {
    auto it = map.find(arg);
    if (it != map.end()) {
      return it->second;
    }
    map.emplace(arg, map.size());
    return map.size() - 1;
  }

 public:
  // These fields map their entires to indices in the corresponding CodeObject
  // table. The index is used as the argument to opcodes.
  // Names
  std::unordered_map<std::string, size_t> names_;
  // Local variables
  std::unordered_map<std::string, size_t> varnames_;
  // Constants
  // TODO make sure this is the right hash fn
  std::unordered_map<Obj, size_t, DictKeyHash> consts_;

  SymbolTableEntry* symbolTableEntry_;

  std::vector<std::unique_ptr<BasicBlock>> basicBlocks_;
  BasicBlock* curBlock_ = nullptr;
};

// Takes a CompilerUnit and produces a CodeObject, ready to be run.
// TODO this is really just a function
class Assembler {
 public:
  static const c10::intrusive_ptr<CodeObject> run(CompilerUnit* compilerUnit) {
    if (!compilerUnit->curBlock_->hasReturn) {
      // Add a none return if there is no explicit return instruction
      auto returnBlock = compilerUnit->addBlock();
      compilerUnit->enterBlock(returnBlock);
      size_t idx = compilerUnit->addConst(Obj());
      compilerUnit->addInstruction(Instruction(OpCode::LOAD_CONST, idx));
      compilerUnit->addInstruction(
          Instruction(OpCode::RETURN_VALUE, /*unused*/ 0));
    }

    // Assemble jump offsets.
    // 1. Compute a reverse post-order traversal
    BasicBlock* root = compilerUnit->basicBlocks_.at(0).get();
    auto orderedBlocks = dfs(root);
    assert(orderedBlocks.size() == compilerUnit->basicBlocks_.size());
    std::reverse(orderedBlocks.begin(), orderedBlocks.end());

    // 2. Compute the pc offset for each block
    size_t totalSize = 0;
    for (BasicBlock* block : orderedBlocks) {
      block->offset = totalSize;
      totalSize += block->instructions_.size();
    }

    // 3. Fix up all the jump instructions to point to the correct offsets.
    for (BasicBlock* block : orderedBlocks) {
      for (auto& instruction : block->instructions_) {
        if (instruction.jumpTarget_) {
          instruction.inst_.arg1 = instruction.jumpTarget_->offset;
        }
      }
    }

    // 4. Emit all the code in a flat instruction list.
    std::vector<Instruction> flattenedInstructions;
    for (BasicBlock* block : orderedBlocks) {
      for (const auto& instruction : block->instructions_) {
        flattenedInstructions.push_back(instruction.inst_);
      }
    }

    // size_t offset = 0;
    // for (const auto& i : flattenedInstructions) {
    //   fmt::print("{}{:<20}\n", i, offset++);
    // }
    auto ret = c10::make_intrusive<CodeObject>();
    // TODO don't copy here
    ret->names = flatten(compilerUnit->names_);
    ret->varnames = flatten(compilerUnit->varnames_);
    ret->constants = flatten(compilerUnit->consts_);
    ret->instructions = flattenedInstructions;
    return ret;
  }

 private:
  static std::vector<BasicBlock*> dfs(BasicBlock* block) {
    std::unordered_set<BasicBlock*> seen;
    std::vector<BasicBlock*> postorder;
    dfsImpl(block, seen, postorder);
    return postorder;
  }
  static void dfsImpl(
      BasicBlock* block,
      std::unordered_set<BasicBlock*>& seen,
      std::vector<BasicBlock*>& postorder) {
    if (seen.count(block)) {
      return;
    }
    seen.insert(block);

    // Descend through the next block in normal control flow
    if (block->next_ != nullptr) {
      dfsImpl(block->next_, seen, postorder);
    }

    // Descend through any blocks that the current block jumps to
    for (const auto& instruction : block->instructions_) {
      if (instruction.jumpTarget_ != nullptr) {
        dfsImpl(instruction.jumpTarget_, seen, postorder);
      }
    }
    postorder.push_back(block);
  }

  template <typename T, typename H, typename E>
  static std::vector<T> flatten(
      const std::unordered_map<T, size_t, H, E>& map) {
    std::vector<T> flattened(map.size());
    for (const auto& pr : map) {
      const T& key = pr.first;
      size_t idx = pr.second;
      flattened[idx] = key;
    }
    return flattened;
  }
};

// AST -> CodeObject
// TODO cleanup usage of `emit` vs. accessing curUnit_.
class Compiler {
 public:
  explicit Compiler(std::unique_ptr<SymbolTable> st)
      : symbolTable_(std::move(st)) {}

  c10::intrusive_ptr<CodeObject> run(Mod module) {
    push(module);
    visit(module.body());
    auto cu = pop();
    return Assembler::run(cu);
  }

  c10::intrusive_ptr<CodeObject> run(Def def) {
    push(def);
    visit(def.statements());
    auto cu = pop();
    return Assembler::run(cu);
  }

 private:
  template <typename T>
  void visit(List<T> elements) {
    for (T element : elements) {
      visit(element);
    }
  }

  void visit(const Def& def) {
    push(def);
    visit(def.statements());
    CompilerUnit* cs = pop();
    auto co = Assembler::run(cs);

    fmt::print("\nBytecode for fn '{}':\n", def.name().name());
    minipy::dump(*co);
    size_t idx = curUnit_->addConst(co);
    emit(Instruction(OpCode::LOAD_CONST, idx));
    idx = curUnit_->addConst(def.name().name());
    emit(Instruction(OpCode::LOAD_CONST, idx));
    emit(Instruction(OpCode::MAKE_FUNCTION, /*useless*/ 0));
    curUnit_->emitNameOp(def.name(), NameContext::Store);
  }

  void visit(const ClassDef& classDef) {
    emit(Instruction(OpCode::LOAD_BUILD_CLASS, /*useless*/ 0));
    push(classDef);
    visit(classDef.body());
    CompilerUnit* cs = pop();

    auto co = Assembler::run(cs);
    size_t idx = curUnit_->addConst(co);
    emit(Instruction(OpCode::LOAD_CONST, idx));
    idx = curUnit_->addConst(classDef.name().name());
    emit(Instruction(OpCode::LOAD_CONST, idx));
    emit(Instruction(OpCode::MAKE_FUNCTION, /*useless*/ 0));

    // Load the name again
    emit(Instruction(OpCode::LOAD_CONST, idx));
    emit(Instruction(OpCode::CALL_FUNCTION, 2));
    curUnit_->emitNameOp(classDef.name(), NameContext::Store);

    fmt::print("\nBytecode for class '{}':\n", classDef.name().name());
    minipy::dump(*co);
  }

  void emit(Instruction instruction) {
    curUnit_->addInstruction(instruction);
  }

  // TODO clean
  void emit(Instruction instruction, BasicBlock* b) {
    curUnit_->addInstruction(instruction, b);
  }

  void visit(const For& for_) {
    auto start = curUnit_->addBlock();
    // TODO we don't support or-else statements yet.
    // auto cleanup = curUnit_->addBlock();
    auto end = curUnit_->addBlock();

    if (for_.itrs().size() > 1) {
      // TODO py AST says there is only one expr, while ours is a list<expr>,
      // need to investigate why there is a difference
      throw ErrorReport(for_.range()) << "can't have more than one iter expr";
    }
    visit(for_.itrs());
    emit(Instruction(OpCode::GET_ITER, /*useless*/ 0));

    // Enter the loop body
    // TODO we need frameblock infrasturcutre to support early
    // returns/continues/breaks
    curUnit_->enterBlock(start);
    emit(Instruction(OpCode::FOR_ITER, /*to be assembled*/ 0), end);
    if (for_.targets().size() > 1) {
      // TODO py AST says there is only one expr, while ours is a list<expr>,
      // need to investigate why there is a difference
      throw ErrorReport(for_.range()) << "can't have more than one iter expr";
    }
    visit(for_.targets());
    visit(for_.body());
    emit(Instruction(OpCode::JUMP_ABSOLUTE, /*to be assembled*/ 0), start);
    curUnit_->enterBlock(end);
  }

  void visit(const AugAssign& augAssign) {
    const auto& lhs = augAssign.lhs();
    switch (lhs.kind()) {
      // TODO  implement
      // case '.': {
      // }
      // case TK_SUBSCRIPT: {
      // }
      case TK_VAR: {
        Ident lhsName = Var(lhs).name();
        curUnit_->emitNameOp(lhsName, NameContext::Load);
        visit(augAssign.rhs());

        switch (augAssign.aug_op()) {
          case '+':
            emit(Instruction(OpCode::INPLACE_ADD, /*unused*/ 0));
            break;
          default:
            throw ErrorReport(augAssign) << "Unknown augmented assignment: "
                                         << kindToString(augAssign.aug_op());
        }
        curUnit_->emitNameOp(lhsName, NameContext::Store);
      } break;
      default:
        throw ErrorReport(lhs) << fmt::format(
            "Invalid node type ({}) for augmented assignment",
            kindToString(lhs.kind()));
    }
  }

  void visit(const Assert& assert) {
    // assert.test()
    auto end = curUnit_->addBlock();
    visit(assert.test());
    emit(Instruction(OpCode::POP_JUMP_IF_TRUE, /*to be assembled*/ 0), end);
    emit(Instruction(OpCode::RAISE_VARARG, /*nyi*/ 0));
    curUnit_->enterBlock(end);
  }

  void visit(const Stmt& stmt) {
    switch (stmt.kind()) {
      case TK_ASSIGN:
        return visit(Assign(stmt));
      case TK_RETURN:
        return visit(Return(stmt));
      case TK_DEF:
        return visit(Def(stmt));
      case TK_EXPR_STMT: {
        visit(ExprStmt(stmt).expr());
        emit(Instruction(OpCode::POP_TOP, /*unused*/ 0));
      } break;
      case TK_CLASS_DEF:
        return visit(ClassDef(stmt));
      case TK_AUG_ASSIGN:
        return visit(AugAssign(stmt));
      case TK_RAISE:
        // TODO actually support raising
        emit(Instruction(OpCode::RAISE_VARARG, /*nyi*/ 0));
        break;
      case TK_ASSERT:
        visit(Assert(stmt));
        break;
      case TK_PASS:
        break;
      case TK_BREAK:
        break;
      case TK_DELETE:
        break;
      case TK_CONTINUE:
        break;
      case TK_IF:
        return visit(If(stmt));
        break;
      case TK_FOR:
        return visit(For(stmt));
        break;
      case TK_WHILE:
        break;
      case TK_GLOBAL:
        break;
      default:
        throw ErrorReport(stmt.range())
            << "compiler: unsupported statemet " << kindToString(stmt.kind());
    }
  }

  void visit(If if_) {
    // Push a new code block
    // Basic block representing the continuation of control flow (after this if
    // statement).
    auto end = curUnit_->addBlock();

    // BasicBlock
    BasicBlock* falseTarget = end;
    bool hasElse = if_.falseBranch().size() > 0;
    if (hasElse) {
      // Add a basic block for the body of the else condition
      falseTarget = curUnit_->addBlock();
    } else {
      falseTarget = end;
    }

    // TODO there is a small optimization you can do skip bytecode emission if
    // the predicate is constant and true/false

    // Handle the jump predication
    // TODO there are also optimizations in compiler_jump_if that may be
    // interesting
    visit(if_.cond());
    emit(
        Instruction(OpCode::POP_JUMP_IF_FALSE, /*to be assembled*/ 0),
        falseTarget);
    visit(if_.trueBranch());
    if (hasElse) {
      // To the end of the if-body, add a jump to the end (to skip the
      // else-body)
      emit(Instruction(OpCode::JUMP_ABSOLUTE, /*to be assembled*/ 0), end);

      // Then enter the else-block and start emitting it
      curUnit_->enterBlock(falseTarget);
      visit(if_.falseBranch());
    }
    curUnit_->enterBlock(end);
  }

  void visit(const Assign& assign) {
    // TODO we ideally have a context for each `Var` to describe whether it's a
    // load or store, so we can do the
    Expr rhs = assign.rhs().get();
    visit(rhs);
    {
      AssignmentContextGuard g(this, NameContext::Store);
      visit(assign.lhs());
    }
  }

  void visit(const List<Expr>& exprs) {
    for (const Expr& expr : exprs) {
      visit(expr);
    }
  }

  void visit(const UnaryOp& op) {
    switch (op.kind()) {
      case TK_NOT:
        visit(op.operand());
        emit(Instruction(OpCode::UNARY_NOT, /*unused*/ 0));
        break;
      case TK_UNARY_MINUS:
      case '~':
        throw ErrorReport(op) << "unsupported unary op";
    }
  }

  void visit(StringLiteral stringLiteral) {
    Obj constant = stringLiteral.text();
    size_t idx = curUnit_->addConst(constant);
    emit(Instruction(OpCode::LOAD_CONST, idx));
  }

  void visit(const Expr& expr) {
    switch (expr.kind()) {
      case TK_VAR:
        return visit(Var(expr));
      case TK_IF_EXPR:
      case TK_AND:
      case TK_OR:
      case '<':
      case '>':
      case TK_IS:
      case TK_ISNOT:
      case TK_EQ:
      case TK_LE:
      case TK_GE:
      case TK_NE:
      case '+':
      case '/':
      case '%':
      case TK_FLOOR_DIV:
        return visit(BinOp(expr));
      case TK_APPLY:
        return visit(Apply(expr));
      case '.':
        return visit(Select(expr));
      case TK_CONST:
        return visit(Const(expr));
      case TK_STRINGLITERAL:
        return visit(StringLiteral(expr));
      case TK_NONE: {
        // TODO: this should probably be a global that we're not putting in
        // every code object
        size_t idx = curUnit_->addConst(Obj());
        emit(Instruction(OpCode::LOAD_CONST, idx));
      } break;
      case TK_UNARY_MINUS:
      case TK_NOT:
      case '~':
        return visit(UnaryOp(expr));
      case TK_LIST_LITERAL: {
        // TODO unpacking and stuff isn't handled
        auto literal = ListLiteral(expr);
        visit(literal.inputs());
        emit(Instruction(OpCode::BUILD_LIST, literal.inputs().size()));
      } break;
      case '-':
      case '*':
      case TK_STARRED:
      case TK_TRUE:
      case TK_FALSE:
      case TK_CAST:
      case TK_SUBSCRIPT:
      case TK_SLICE_EXPR:
      case TK_TUPLE_LITERAL:
      case TK_DICT_LITERAL:
      case '@':
      case TK_POW:
      case TK_LSHIFT:
      case TK_RSHIFT:
      case '&':
      case '^':
      case '|':
      case TK_LIST_COMP:
      case TK_DOTS:
      case TK_IN:
        throw ErrorReport(expr)
            << "compiler: unsupported expr " << kindToString(expr.kind());
    }
  }

  void visit(const Select& select) {
    {
      // The stuff on the lhs of selection operation is always a load
      // TODO is this true?
      AssignmentContextGuard g(this, NameContext::Load);
      visit(select.value());
    }
    const std::string& attrName = select.selector().name();
    size_t idx = curUnit_->addName(attrName);
    OpCode op;
    switch (assignmentContext_) {
      case NameContext::Load:
        op = OpCode::LOAD_ATTR;
        break;
      case NameContext::Store:
        op = OpCode::STORE_ATTR;
        break;
      default:
        throw ErrorReport(select) << "Unknown name context for select";
    }
    emit(Instruction(op, idx));
  }

  void visit(const Const& const_) {
    Obj constant;
    if (const_.isFloatingPoint()) {
      constant = const_.asFloatingPoint();
    } else {
      constant = const_.asIntegral();
    }
    size_t idx = curUnit_->addConst(constant);
    emit(Instruction(OpCode::LOAD_CONST, idx));
  }

  static Instruction getInstructionForBinaryOperator(const BinOp binOp) {
    switch (binOp.kind()) {
      case TK_AND:
        return Instruction(
            OpCode::JUMP_IF_FALSE_OR_POP, /*to be assembled=0*/ 0);
      case TK_OR:
        return Instruction(
            OpCode::JUMP_IF_TRUE_OR_POP, /*to be assembled=0*/ 0);
      case '<':
        return Instruction(
            OpCode::COMPARE_OP, static_cast<int>(CompareOp::LessThan));
      case '>':
        return Instruction(
            OpCode::COMPARE_OP, static_cast<int>(CompareOp::GreaterThan));
      case TK_IS:
        return Instruction(OpCode::COMPARE_OP, static_cast<int>(CompareOp::Is));
      case TK_ISNOT:
        return Instruction(
            OpCode::COMPARE_OP, static_cast<int>(CompareOp::IsNot));
      case TK_EQ:
        return Instruction(
            OpCode::COMPARE_OP, static_cast<int>(CompareOp::Equal));
      case TK_LE:
        return Instruction(
            OpCode::COMPARE_OP, static_cast<int>(CompareOp::LessThanEqual));
      case TK_GE:
        return Instruction(
            OpCode::COMPARE_OP, static_cast<int>(CompareOp::GreaterThanEqual));
      case TK_NE:
        return Instruction(
            OpCode::COMPARE_OP, static_cast<int>(CompareOp::NotEqual));
      case '+':
        return Instruction(OpCode::BINARY_ADD, /*unused*/ 0);
      case '/':
        return Instruction(OpCode::BINARY_TRUE_DIVIDE, /*unused*/ 0);
      case '%':
        return Instruction(OpCode::BINARY_MODULO, /*unused*/ 0);
      case TK_FLOOR_DIV:
        return Instruction(OpCode::BINARY_FLOOR_DIVIDE, /*unused*/ 0);
      case '*':
      case '-':
      case '@':
      case TK_POW:
      case TK_LSHIFT:
      case TK_RSHIFT:
      case '&':
      case '^':
      case '|':
      case TK_IN:
      default:
        throw ErrorReport(binOp) << fmt::format(
            "Unhandled binary operator: {}", kindToString(binOp.kind()));
    }
  }

  void visit(BinOp binOp) {
    // TODO make sure binary op handling distinguishes something like
    // x < 4 < 3
    //  from
    // (x < 4) < 3
    // TODO lots of code duplication
    Instruction instruction = getInstructionForBinaryOperator(binOp);
    if (binOp.kind() == TK_AND || binOp.kind() == TK_OR) {
      // `and` and `or` are handled differently from other binary operators,
      // since they have short-circuiting behavior
      auto end = curUnit_->addBlock();
      visit(binOp.lhs());
      emit(instruction, end);
      visit(binOp.rhs());
      curUnit_->enterBlock(end);
    } else {
      visit(binOp.lhs());
      visit(binOp.rhs());
      emit(instruction);
    }
  }

  void visit(const Return& ret) {
    // TODO handle returning none, early returns, etc.
    visit(ret.expr());
    emit(Instruction(OpCode::RETURN_VALUE, 0));
  }

  // std::vector<std::string> methods;

  void visit(const Apply& apply) {
    // // TODO this is very similar to maybe_optimize_method_call .
    // // Need to figure out how to make this more generic, it only works on
    // // something that looks like `self.foo`.
    // if (apply.callee().kind() == '.') {
    //   auto s = Select(apply.callee());
    //   if (s.value().kind() == TK_VAR &&
    //       Var(s.value()).name().name() == "self") {
    //     // This is a method call on self.
    //     fmt::print("called method: {}\n", s.selector().name());
    //     methods.push_back(s.selector().name());
    //   }
    // }

    visit(apply.callee());
    visit(apply.inputs());
    emit(Instruction(OpCode::CALL_FUNCTION, apply.inputs().size()));
  }

  void visit(const Var& var) {
    curUnit_->emitNameOp(var.name(), assignmentContext_);
  }

  // Push a new compiler unit on to the stack
  void push(TreeRef ref) {
    auto symbolTableEntry = symbolTable_->lookup(ref);
    compilerUnits_.push_back(std::make_unique<CompilerUnit>(symbolTableEntry));
    curUnit_ = compilerUnits_.back().get();
    stack_.push_back(curUnit_);
  }

  CompilerUnit* pop() {
    assert(!stack_.empty());
    CompilerUnit* back = stack_.back();
    stack_.pop_back();
    if (stack_.empty()) {
      curUnit_ = nullptr;
    } else {
      curUnit_ = stack_.back();
    }
    return back;
  }

  // Used to determine whether we should add symbols in the context of an
  // assignment (e.g. as a STORE and not a LOAD).
  // TODO duplicated with the one in symbol table
  // TODO this should be computed as part of the AST
  NameContext assignmentContext_ = NameContext::Load;
  class AssignmentContextGuard {
   public:
    AssignmentContextGuard(Compiler* builder, NameContext assignmentContext)
        : builder_(builder) {
      prev_ = builder->assignmentContext_;
      builder_->assignmentContext_ = assignmentContext;
    }
    ~AssignmentContextGuard() noexcept {
      builder_->assignmentContext_ = prev_;
    }

   private:
    Compiler* builder_;
    NameContext prev_;
  };

  std::vector<std::unique_ptr<CompilerUnit>> compilerUnits_;
  std::vector<CompilerUnit*> stack_;
  CompilerUnit* curUnit_ = nullptr;
  std::unique_ptr<SymbolTable> symbolTable_;
};

} // namespace

c10::intrusive_ptr<CodeObject> emit(Mod mod) {
  auto st = SymbolTable::build(mod);
  Compiler compiler(std::move(st));
  return compiler.run(mod);

  // auto co = compiler.run(mod);
  // fmt::print("\nBytecode for module:\n");
  // co->dump();
  // // return co;

  // IValue torch = c10::make_intrusive<TorchModule>();
  // IValue print = c10::make_intrusive<BuiltinFunction>("print", [](IValue
  // args) {
  //   std::cout << args << std::endl;
  //   return IValue();
  // });

  // FrameObject frame;
  // frame.globals.emplace("torch", torch);
  // frame.globals.emplace("print", print);
  // frame.fastLocals.reserve(co->varnames.size());
  // // fmt::print("size: {}\n", co->varnames.size());
  // // frame.locals[0] = torch::ones({2, 2});

  // Interpreter interpreter(*co, frame);
  // IValue ret = interpreter.run();

  // fmt::print("New returned: {}\n", ret);
  // return co;

  // CompilationUnit cu;
  // cu.define(c10::nullopt, {def}, {nativeResolver()}, nullptr, false);
  // auto ret2 = cu.run_method(def.name().name(), torch::ones({2, 2}));
  // fmt::print("Old returned: {}\n", ret2);
}

std::string gatherName(const CodeObject& codeObject, size_t curIdx) {
  const auto& instructions = codeObject.instructions;
  assert(curIdx < instructions.size());
  std::vector<std::string> name;
  const auto& startingInstruction = instructions[curIdx];
  if (startingInstruction.op == OpCode::LOAD_FAST) {
    name.push_back(codeObject.varnames.at(instructions[curIdx].arg1));
  } else if (startingInstruction.op == OpCode::LOAD_GLOBAL) {
    name.push_back(codeObject.names.at(instructions[curIdx].arg1));
  } else {
    assert(false);
    //
    // "unexpected copcode for gathername: ",
    // toString(startingInstruction.op));
  }

  while (++curIdx < instructions.size()) {
    const auto& instruction = instructions[curIdx];
    if (instruction.op == OpCode::LOAD_ATTR ||
        instruction.op == OpCode::LOAD_METHOD) {
      name.push_back(codeObject.names.at(instruction.arg1));
    } else {
      break;
    }
  }
  return fmt::format("{}", fmt::join(name, "."));
}

std::vector<std::string> gatherInstanceAttributes(
    const CodeObject& codeObject) {
  minipy::dump(codeObject);
  const auto& instructions = codeObject.instructions;
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  for (size_t i = 0; i < instructions.size(); ++i) {
    if (instructions[i].op == OpCode::LOAD_FAST && instructions[i].arg1 == 0) {
      auto name = gatherName(codeObject, i);
      if (seen.count(name)) {
        continue;
      }
      names.push_back(std::move(name));
    }
  }
  return names;
}

// TODO should return ordered set for determinism reasons
std::vector<std::string> gatherGlobals(const CodeObject& codeObject) {
  const auto& instructions = codeObject.instructions;
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  for (size_t i = 0; i < instructions.size(); ++i) {
    if (instructions[i].op == OpCode::LOAD_GLOBAL) {
      auto name = gatherName(codeObject, i);
      if (seen.count(name)) {
        continue;
      }
      names.push_back(std::move(name));
    }
  }
  return names;
}

c10::intrusive_ptr<CodeObject> emit(Def def) {
  auto st = SymbolTable::build(def);
  Compiler compiler(std::move(st));
  return compiler.run(def);
}
} // namespace minipy
