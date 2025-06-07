use tpde::test_ir::TestIR;

fn main() {
    let tir = r#"
simple(%a) {
entry:
  %const =
  jump ^loop
loop:
  %b = phi [^entry, %const], [^loop, %e]
  %c = phi [^entry, %a], [^loop, %f]
  %d = %c
  %e = %b, %d
  %f = %c
  jump ^loop, ^ret
ret:
  %ret = %b
  terminate
}
"#;
    
    match TestIR::parse(tir) {
        Ok(ir) => {
            println!("Parse successful!");
            println!("{}", ir.print());
        }
        Err(e) => {
            println!("Parse error: {e}");
        }
    }
}