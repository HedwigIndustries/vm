## Virtual machine

This virtual machine executes Python bytecode in Python :)

The virtual machine implementation is in the `vm.py` file.

### Notes

* The project does not contain complete specifications for the all operation of the virtual machine; there are bugs.
  This is a toy implementation.
* All the necessary commands for the interpreter to work correctly are in the file `vm_cscorer.py`. 
* Commands grouped by complexity levels. In this implementation, operations 1,2,3 priority are implemented.

### Tests

#### How to run all tests

```bash
$ pytest test_public.py -vvv --tb=no
```

#### How to run a specific test case

```bash
$ pytest test_public.py::test_all_cases[simple] -vvv
```