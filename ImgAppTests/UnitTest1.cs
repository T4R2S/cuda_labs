using NUnit.Framework;

namespace ImgAppTests
{
    [TestFixture]
    public class UnitTest1
    {
        [Test]
        public void TestMethod1()
        {
            //arrange
            var a = 4;
            var b = 6;

            //act
            var c = a + b;

            //assert
            Assert.AreEqual(10, c);
        }
    }
}
