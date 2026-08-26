/**
 * Proper math typesetting for the φ_asym symbol — an italic phi with an "asym"
 * subscript, rendered in a serif math font. Use instead of the literal text
 * "φ_asym" anywhere it appears in the UI.
 */
interface PhiAsymProps {
  className?: string;
}

const MATH_FONT = '"Cambria Math", "STIX Two Math", "Latin Modern Math", Georgia, "Times New Roman", serif';

export function PhiAsym({ className }: PhiAsymProps) {
  return (
    <span className={className} style={{ fontFamily: MATH_FONT, fontStyle: 'italic', whiteSpace: 'nowrap' }}>
      φ
      <sub style={{ fontSize: '0.65em', fontStyle: 'italic' }}>asym</sub>
    </span>
  );
}
