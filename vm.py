"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp
import operator


def arg_binging(code: types.CodeType,
                def_args: tuple[tp.Any, ...] | None,
                def_kwargs: dict[str, tp.Any],
                *args: tp.Any,
                **kwargs: tp.Any) -> dict[str, tp.Any]:
    # broken arg-binding, need some fix
    var_names = code.co_varnames
    arg_count = code.co_argcount
    know_count = code.co_kwonlyargcount
    result = {}

    for i in range(arg_count):
        if i < len(args):
            result[var_names[i]] = args[i]
        if var_names[i] in kwargs:
            result[var_names[i]] = kwargs[var_names[i]]
        else:
            if def_args is not None and (i - len(result) >= 0):
                result[var_names[i]] = def_args[i - len(result)]

    for i in range(arg_count, arg_count + know_count):
        if var_names[i] in kwargs:
            result[var_names[i]] = kwargs[var_names[i]]
        elif var_names[i] in def_kwargs:
            result[var_names[i]] = def_kwargs[var_names[i]]

    return result


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.11/Include/frameobject.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """

    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.UNDEFINED: str = "UNDEFINED"
        self.FORWARD: str = "FORWARD"
        self.BACKWARD: str = "BACKWARD"
        self.NULL: str = "NULL"
        self.target_delta: int = 0
        self.current_offset: int = 0
        self.kill: bool = False
        self.last_exception = BaseException()
        self.jump: str = self.UNDEFINED
        self.code: types.CodeType = frame_code
        self.builtins: dict[str, tp.Any] = frame_builtins
        self.globals: dict[str, tp.Any] = frame_globals
        self.locals: dict[str, tp.Any] = frame_locals
        self.data_stack: tp.Any = []
        self.return_value: tp.Any = None

    BINARY_OPERATIONS: dict[int, tp.Callable[[tp.Any, tp.Any], tp.Any]] = {
        0: operator.add,
        10: operator.sub,
        5: operator.mul,
        11: operator.truediv,
        2: operator.floordiv,
        6: operator.mod,
        4: operator.matmul,
        8: operator.pow,
        3: operator.lshift,
        9: operator.rshift,
        1: operator.and_,
        7: operator.or_,
        12: operator.xor,
        13: operator.iadd,
        23: operator.isub,
        18: operator.imul,
        24: operator.itruediv,
        15: operator.ifloordiv,
        19: operator.imod,
        17: operator.imatmul,
        21: operator.ipow,
        16: operator.ilshift,
        22: operator.irshift,
        14: operator.iand,
        20: operator.ior,
        25: operator.ixor
    }

    COMPARE: dict[str, tp.Any] = {
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        ">=": operator.ge
    }

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        pointer: int = 0
        self.current_offset = 0
        instructions = list(dis.get_instructions(self.code))
        while pointer < len(instructions):
            if self.jump == self.FORWARD:
                while instructions[pointer].offset != self.target_delta:
                    pointer += 1
                self.jump = self.UNDEFINED

            if self.jump == self.BACKWARD:
                while instructions[pointer].offset != self.target_delta:
                    pointer -= 1
                self.jump = self.UNDEFINED

            instruction = instructions[pointer]
            self.current_offset = instruction.offset

            if self.kill:
                break

            if self.NULL in instruction.argrepr:
                self.push(None)

            getattr(self, instruction.opname.lower() + "_op")(instruction.argval)
            pointer += 1
        return self.return_value

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self, _: int) -> None:
        self.push(None)

    def precall_op(self, arg: int) -> tp.Any:
        pass

    def call_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-CALL
        """
        arguments = self.popn(arg)
        callable_or_obj = self.pop()
        method_or_null = self.pop()
        if method_or_null is None:
            self.push(callable_or_obj(*arguments))
        else:
            self.push(method_or_null(callable_or_obj, *arguments))

    def pop_top_op(self, _: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    def make_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-MAKE_FUNCTION
        """
        code = self.pop()  # the code associated with the function (at TOS1)
        if arg & 8:
            self.pop()
        if arg & 4:
            self.pop()
        def_kwargs: dict[str, tp.Any] = {}
        if arg & 2:
            def_kwargs = self.pop()
        def_args = None
        if arg & 1:
            def_args = tuple(self.pop())

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            parsed_args: dict[str, tp.Any] = arg_binging(code, def_args, def_kwargs, *args,
                                                         **kwargs)
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)
            frame = Frame(code, self.builtins, self.globals, f_locals)
            return frame.run()

        self.push(f)

    def load_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        if arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            self.last_exception = NameError()
            raise NameError()

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_NAME
        """
        if arg in self.locals:
            self.push(self.locals[arg])
        else:
            self.load_global_op(arg)

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(arg)

    def load_attr_op(self, name: str) -> None:
        obj = self.pop()
        attr = getattr(obj, name)
        self.push(attr)

    def load_fast_op(self, name: str) -> None:
        if name in self.locals:
            self.push(self.locals[name])
        else:
            self.last_exception = UnboundLocalError()
            raise UnboundLocalError()

    def load_assertion_error_op(self, _: tp.Any) -> None:
        self.push(AssertionError)

    def load_build_class_op(self, _: tp.Any) -> None:
        self.push(__build_class__)

    def load_method_op(self, name: str) -> None:
        obj = self.pop()
        method = getattr(obj, name)
        if isinstance(method, staticmethod) or callable(method):
            self.push(getattr(type(obj), name))
            self.push(obj)
        else:
            self.push(None)
            self.push(method)

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_NAME
        """
        self.locals[arg] = self.pop()

    def store_global_op(self, name: str) -> None:
        self.globals[name] = self.pop()

    def store_attr_op(self, name: str) -> None:
        value, obj = self.popn(2)
        setattr(obj, name, value)

    def store_fast_op(self, name: str) -> None:
        self.locals[name] = self.pop()

    def store_subscr_op(self, _: tp.Any) -> None:
        value, collection, key = self.popn(3)
        collection[key] = value

    def nop_op(self, arg: tp.Any) -> None:
        pass

    def pop_expect_op(self, _: tp.Any) -> None:
        self.pop()

    def return_value_op(self, _: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.kill = True
        self.return_value = self.pop()

    def binary_op_op(self, op: int) -> None:
        left_operand, right_operand = self.popn(2)
        operation = self.BINARY_OPERATIONS[op]
        result = operation(left_operand, right_operand)
        self.push(result)

    def binary_subscr_op(self, _: tp.Any) -> None:
        collection, key = self.popn(2)
        self.push(collection[key])

    def unary_invert_op(self, _: tp.Any) -> None:
        self.push(~(self.pop()))

    def unary_negative_op(self, _: tp.Any) -> None:
        self.push(-(self.pop()))

    def unary_not_op(self, _: tp.Any) -> None:
        self.push(not (self.pop()))

    def unary_positive_op(self, _: tp.Any) -> None:
        self.push(+(self.pop()))

    def compare_op_op(self, op: str) -> None:
        left, right = self.popn(2)
        result = self.COMPARE[op](left, right)
        self.push(result)

    def set_jump(self, target_delta: int, jump_type: str) -> None:
        self.target_delta = target_delta
        self.jump = jump_type

    def jump_forward_op(self, target_delta: int) -> None:
        self.set_jump(target_delta, self.FORWARD)

    def pop_jump_forward_if_false_op(self, target_delta: int) -> None:
        condition: bool = self.pop()
        if not condition:
            self.set_jump(target_delta, self.FORWARD)

    def pop_jump_forward_if_none_op(self, target_delta: int) -> None:
        item = self.pop()
        if item is None:
            self.set_jump(target_delta, self.FORWARD)

    def pop_jump_forward_if_not_none_op(self, target_delta: int) -> None:
        item = self.pop()
        if item is not None:
            self.set_jump(target_delta, self.FORWARD)

    def pop_jump_forward_if_true_op(self, target_delta: int) -> None:
        condition: bool = self.pop()
        if condition:
            self.set_jump(target_delta, self.FORWARD)

    def jump_backward_op(self, target_delta: int) -> None:
        self.set_jump(target_delta, self.BACKWARD)

    def pop_jump_backward_if_false_op(self, target_delta: int) -> None:
        condition: bool = self.pop()
        if not condition:
            self.set_jump(target_delta, self.BACKWARD)

    def pop_jump_backward_if_none_op(self, target_delta: int) -> None:
        item = self.pop()
        if item is None:
            self.set_jump(target_delta, self.BACKWARD)

    def pop_jump_backward_if_not_none_op(self, target_delta: int) -> None:
        item = self.pop()
        if item is not None:
            self.set_jump(target_delta, self.BACKWARD)

    def pop_jump_backward_if_true_op(self, target_delta: int) -> None:
        condition: bool = self.pop()
        if condition:
            self.set_jump(target_delta, self.BACKWARD)

    def undefined_jump(self, target_delta: int) -> None:
        if self.current_offset < target_delta:
            self.set_jump(target_delta, self.FORWARD)
        else:
            self.set_jump(target_delta, self.BACKWARD)

    def jump_if_false_or_pop_op(self, target_delta: int) -> None:
        if self.top() is False:
            self.undefined_jump(target_delta)
        else:
            self.pop()

    def jump_if_true_or_pop_op(self, target_delta: int) -> None:
        if self.top() is True:
            self.undefined_jump(target_delta)
        else:
            self.pop()

    def for_iter_op(self, target_delta: int) -> None:
        iterator = self.top()
        try:
            self.push(next(iterator))
        except StopIteration:
            self.pop()
            self.set_jump(target_delta, self.FORWARD)

    def get_iter_op(self, _: tp.Any) -> None:
        iter_obj = self.pop()
        self.push(iter(iter_obj))

    def get_len_op(self, _: tp.Any) -> None:
        self.push(len(self.top()))

    def list_extend_op(self, i: int) -> None:
        elem = self.pop()
        lst: list[tp.Any] = self.data_stack[-i]
        lst.extend(elem)

    def build_list_op(self, count: int) -> None:
        self.push(self.popn(count))

    def build_tuple_op(self, count: int) -> None:
        self.push(tuple(self.popn(count)))

    def list_to_tuple_op(self, _: tp.Any) -> None:
        self.push(tuple(self.pop()))

    def build_set_op(self, count: int) -> None:
        self.push(set(self.popn(count)))

    def set_update_op(self, i: int) -> None:
        elem = self.pop()
        st: set[tp.Any] = self.data_stack[-i]
        st.update(elem)

    def build_slice_op(self, argc: int) -> None:
        step = self.pop() if argc == 3 else 1
        start, end = self.popn(2)
        self.push(slice(start, end, step))

    def unpack_sequence_op(self, count: int) -> None:
        sequence = self.pop()
        if len(sequence) == count:
            for item in reversed(sequence):
                self.push(item)
        else:
            self.last_exception = ValueError()
            raise ValueError()

    def dict_update_op(self, i: int) -> None:
        pair = self.pop()
        self.data_stack[-i].update(pair)

    def build_const_key_map_op(self, count: int) -> None:
        key_tuple = self.pop()
        values = self.popn(count)
        res_map = {}
        for i in range(count):
            res_map[key_tuple[i]] = values[i]
        self.push(res_map)

    def build_map_op(self, count: int) -> None:
        result = {}
        for i in range(count):
            value = self.pop()
            key = self.pop()
            result[key] = value
        self.push(result)

    def build_string_op(self, count: int) -> None:
        strings = list(map(lambda x: str(x), self.popn(count)))
        self.push("".join(strings))

    def format_value_op(self, frmt: str) -> None:
        if frmt == "":
            self.push(f'{self.pop()}')
        elif frmt == "str":
            self.push(f'{str(self.pop())}')
        elif frmt == "repl":
            self.push(f'{repr(self.pop())}')
        elif frmt == "ascii":
            self.push(f'{ascii(self.pop())}')

    def delete_global_op(self, var_num: str) -> None:
        del self.globals[var_num]

    def delete_fast_op(self, var_num: str) -> None:
        del self.locals[var_num]

    def delete_name_op(self, name: str) -> None:
        if name in self.locals:
            del self.locals[name]
        elif name in self.globals:
            del self.globals[name]

    def delete_attr_op(self, name: str) -> None:
        obj = self.pop()
        delattr(obj, name)

    def delete_subscr_op(self, _: tp.Any) -> None:
        collection, key = self.popn(2)
        del collection[key]

    def is_op_op(self, invert: int) -> None:
        left, right = self.popn(2)
        self.push(left is not right if invert == 1 else left is right)

    def contains_op_op(self, invert: int) -> None:
        query, container = self.popn(2)
        self.push(not (query in container) if invert == 1 else (query in container))

    def copy_op(self, i: int) -> None:
        item = self.data_stack[-i]
        self.push(item)

    def swap_op(self, i: int) -> None:
        first = self.pop()
        items = self.popn(i - 2)
        last = self.pop()
        self.push(first)
        for item in items:
            self.push(item)
        self.push(last)

    def pop_expect(self, _: tp.Any) -> None:
        self.pop()

    def check_exc_mathc_op(self, _: tp.Any) -> None:
        e, e_type = self.popn(2)
        self.push(e)
        self.push(isinstance(type(e), e_type))

    def raise_varargs_op(self, argc: int) -> None:
        if argc == 0:
            raise self.last_exception
        elif argc == 1:
            e = self.pop()
            exception = e if isinstance(e, Exception) else e()
        else:
            cause = self.pop()
            e = self.pop()
            exception = e if isinstance(e, Exception) else e()
            exception.__cause__ = cause()

        self.last_exception = exception
        raise exception

    def import_name_op(self, name: str) -> None:
        level, from_list = self.popn(2)
        self.push(__import__(name, self.globals, self.locals, from_list, level=level))

    def import_from_op(self, name: str) -> None:
        module = self.top()
        self.push(getattr(module, name))

    def import_star_op(self, _: tp.Any) -> None:
        module_value = self.pop()
        for item in dir(module_value):
            if item[0] != '_':
                self.locals[item] = getattr(module_value, item)


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
