// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract Student {
    struct StudentStruct {
        uint ID;
        string fName;
        string lName;
        uint marks;
    }

    address public owner;
    uint public stdCount;
    mapping(uint => StudentStruct) private stdRecords;

    event RecordAdded(uint ID, string fName, string lName, uint marks);

    modifier onlyOwner() {
        require(msg.sender == owner, "Access denied: Only owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function addNewRecord(
        uint _ID,
        string memory _fName,
        string memory _lName,
        uint _marks
    ) public onlyOwner {
        require(_marks <= 100, "Marks must be between 0 and 100");
        require(stdRecords[_ID].ID == 0, "Student ID already exists");

        stdCount++;
        stdRecords[_ID] = StudentStruct(_ID, _fName, _lName, _marks);

        emit RecordAdded(_ID, _fName, _lName, _marks);
    }

    function getRecord(uint _ID) public view returns (StudentStruct memory) {
        require(stdRecords[_ID].ID != 0, "Record not found");
        return stdRecords[_ID];
    }

    function getTotalStudents() public view returns (uint) {
        return stdCount;
    }

    // Remove payable since we don't need Ether
    fallback() external {
        revert("Function does not exist");
    }

    receive() external payable {
        revert("This contract does not accept Ether");
    }
}